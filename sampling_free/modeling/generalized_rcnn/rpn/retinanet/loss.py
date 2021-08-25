import torch

from sampling_free.layers import smooth_l1_loss, SigmoidFocalLoss, CELoss
from sampling_free.structures import cat_boxlist, boxlist_iou

from ...functions import Matcher, GuidedLoss
from ..utils import concat_box_prediction_layers

class RetinaNetLossComputation(torch.nn.Module):
    def __init__(self, cfg, box_coder, generate_labels_func):
        super().__init__()
        self.proposal_matcher = Matcher(
            cfg.MODEL.RETINANET.FG_IOU_THRESHOLD,
            cfg.MODEL.RETINANET.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
        self.box_coder = box_coder
        self.sampling_free = cfg.MODEL.SAMPLING_FREE
        if self.sampling_free:
            self.guided_loss = GuidedLoss(2)
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
        self.num_classes = cfg.MODEL.RETINANET.NUM_CLASSES
    
    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs, _ = self.proposal_matcher(match_quality_matrix)
        target = target.copy_with_fields(copied_fields)
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

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
            cls_loss (Tensor)
            retinanet_regression_loss (Tensor)
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)

        N = len(labels)
        box_cls, box_regression = concat_box_prediction_layers(box_cls, box_regression)

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        pos_inds = torch.nonzero(labels > 0).squeeze(1)
        pos_numel = pos_inds.numel()

        loc_loss = smooth_l1_loss(
            box_regression[pos_inds],
            regression_targets[pos_inds],
            beta=self.bbox_reg_beta,
            size_average=False,
        ) / max(1, pos_numel * self.regress_norm)

        cls_loss = self.box_cls_loss_func(
            box_cls, labels.int()
        ) / max(1, pos_numel)

        if self.sampling_free:
            return self.guided_loss([cls_loss, loc_loss], 
                ["cls_loss", "loc_loss"])
        else:
            return dict(cls_loss=cls_loss, loc_loss=loc_loss)

def make_retinanet_loss_evaluator(cfg, box_coder):
    loss_evaluator = RetinaNetLossComputation(
        cfg, box_coder, lambda x:x.get_field("labels")
    )
    return loss_evaluator
