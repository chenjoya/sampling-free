"""
This file contains specific functions for computing losses on the RetinaNet
file
"""

import torch
from torch.nn import functional as F

from ..utils import concat_box_prediction_layers

from sampling_free.layers import smooth_l1_loss
from sampling_free.layers import SigmoidFocalLoss, CELoss
from sampling_free.modeling.matcher import Matcher
from sampling_free.modeling.utils import cat
from sampling_free.structures.boxlist_ops import boxlist_iou
from sampling_free.structures.boxlist_ops import cat_boxlist
from sampling_free.modeling.rpn.loss import RPNLossComputation

class RetinaNetLossComputation(RPNLossComputation):
    """
    This class computes the RetinaNet loss.
    """

    def __init__(self, cfg, box_coder, generate_labels_func):
        """
        Arguments:
            proposal_matcher (Matcher)
            box_coder (BoxCoder)
        """
        super().__init__(cfg, box_coder, generate_labels_func)
        self.proposal_matcher = Matcher(
            cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
            cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
        self.box_coder = box_coder
        self.sampling_free = cfg.MODEL.SAMPLING_FREE
        if self.sampling_free:
            self.delta = torch.nn.Parameter(torch.ones(1))
            self.box_cls_loss_func = CELoss()
        else:
            self.box_cls_loss_func =  SigmoidFocalLoss(
                cfg.MODEL.RETINANET.LOSS_GAMMA,
                cfg.MODEL.RETINANET.LOSS_ALPHA
            )
        self.bbox_reg_beta = cfg.MODEL.RETINANET.BBOX_REG_BETA
        self.copied_fields = ['labels']
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['between_thresholds']
        self.regress_norm = cfg.MODEL.RETINANET.BBOX_REG_WEIGHT

    def prepare_targets(self, anchors, targets):
        labels = []
        regression_targets = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            matched_targets = self.match_targets_to_anchors(
                anchors_per_image, targets_per_image, self.copied_fields
            )

            matched_idxs = matched_targets.get_field("matched_idxs")
            labels_per_image = self.generate_labels_func(matched_targets)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0

            # discard anchors that go out of the boundaries of the image
            if "not_visibility" in self.discard_cases:
                labels_per_image[~anchors_per_image.get_field("visibility")] = -1

            # discard indices that are between thresholds
            if "between_thresholds" in self.discard_cases:
                inds_to_discard = matched_idxs == Matcher.BETWEEN_THRESHOLDS
                labels_per_image[inds_to_discard] = -1

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, anchors_per_image.bbox
            )

            labels.append(labels_per_image)
            regression_targets.append(regression_targets_per_image)

        return labels, regression_targets

    def __call__(self, anchors, box_cls, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            retinanet_cls_loss (Tensor)
            retinanet_regression_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = \
                concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)

        retinanet_regression_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / (max(1, pos_inds.numel() * self.regress_norm))
        
        retinanet_cls_loss = self.box_cls_loss_func(
            box_cls,
            labels.int()
        ) / pos_inds.numel()
        
        if self.sampling_free:
            with torch.no_grad():
                r = retinanet_regression_loss / retinanet_cls_loss
            retinanet_cls_loss = r * retinanet_cls_loss

            delta = self.delta ** 2
            retinanet_cls_loss = delta * retinanet_cls_loss
            norm = -delta.log()
            
            return dict(retinanet_cls_loss=retinanet_cls_loss, 
                retinanet_regression_loss=retinanet_regression_loss, norm=norm)
        
        return dict(retinanet_cls_loss=retinanet_cls_loss, 
                retinanet_regression_loss=retinanet_regression_loss)

def generate_retinanet_labels(matched_targets):
    labels_per_image = matched_targets.get_field("labels")
    return labels_per_image

def make_retinanet_loss_evaluator(cfg, box_coder):
    loss_evaluator = RetinaNetLossComputation(
        cfg, box_coder, generate_retinanet_labels
    )
    return loss_evaluator
