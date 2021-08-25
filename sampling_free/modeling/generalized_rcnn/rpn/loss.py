import torch
from torch.functional import F

from sampling_free.layers import smooth_l1_loss, CELoss
from sampling_free.structures import boxlist_iou, cat_boxlist

from .utils import concat_box_prediction_layers
from ..functions import BalancedPositiveNegativeSampler, Matcher

class RPNLossComputation(torch.nn.Module):
    """
    This class computes the RPN loss.
    """

    def __init__(self, cfg, box_coder, generate_labels_func):
        super().__init__()
        self.sampling_free = cfg.MODEL.SAMPLING_FREE
        if self.sampling_free:
            self.ce_loss = CELoss()
        else:
            self.fg_bg_sampler = BalancedPositiveNegativeSampler(
                cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, cfg.MODEL.RPN.POSITIVE_FRACTION
            )
        self.proposal_matcher = Matcher(
            cfg.MODEL.RPN.FG_IOU_THRESHOLD,
            cfg.MODEL.RPN.BG_IOU_THRESHOLD,
            allow_low_quality_matches=True,
        )
        
        self.box_coder = box_coder
        self.copied_fields = []
        self.generate_labels_func = generate_labels_func
        self.discard_cases = ['not_visibility', 'between_thresholds']

    def match_targets_to_anchors(self, anchor, target, copied_fields=[]):
        match_quality_matrix = boxlist_iou(target, anchor)
        matched_idxs, _ = self.proposal_matcher(match_quality_matrix)
        # RPN doesn't need any fields from target
        # for creating the labels, so clear them all
        target = target.copy_with_fields(copied_fields)
        # get the targets corresponding GT for each anchor
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
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
            labels_per_image = self.generate_labels_func(matched_idxs)
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            #print(labels_per_image, labels_per_image.shape, labels_per_image.sum())
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


    def __call__(self, anchors, objectness, box_regression, targets):
        """
        Arguments:
            anchors (list[BoxList])
            objectness (list[Tensor])
            box_regression (list[Tensor])
            targets (list[BoxList])

        Returns:
            rpn_obj_loss (Tensor)
            box_loss (Tensor
        """
        anchors = [cat_boxlist(anchors_per_image) for anchors_per_image in anchors]
        labels, regression_targets = self.prepare_targets(anchors, targets)
        if not self.sampling_free:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
            sampled_pos_inds = torch.nonzero(torch.cat(sampled_pos_inds, dim=0)).squeeze(1)
            sampled_neg_inds = torch.nonzero(torch.cat(sampled_neg_inds, dim=0)).squeeze(1)
            sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        
        objectness, box_regression = \
                concat_box_prediction_layers(objectness, box_regression)
        
        objectness = objectness.squeeze()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        
        if self.sampling_free:
            positive, valid = labels > 0, labels >= 0
            rpn_loc_loss = 0.5 * smooth_l1_loss(
                box_regression[positive],
                regression_targets[positive],
                beta=1.0 / 9,
                size_average=True,
            ) 

            rpn_obj_loss = self.ce_loss(objectness[valid].view(-1,1), 
                labels[valid].int().view(-1, 1)) / positive.sum()
            
            with torch.no_grad():
                ratio = rpn_loc_loss / rpn_obj_loss
            rpn_obj_loss = ratio * rpn_obj_loss
    
        else:
            rpn_loc_loss = smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0 / 9,
                size_average=False,
            ) / (sampled_inds.numel())
            
            rpn_obj_loss = F.binary_cross_entropy_with_logits(
                objectness[sampled_inds], labels[sampled_inds]
            )

        return dict(rpn_obj_loss=rpn_obj_loss, rpn_loc_loss=rpn_loc_loss)

def make_rpn_loss_evaluator(cfg, box_coder):
    loss_evaluator = RPNLossComputation(
        cfg, box_coder, 
        lambda x:x>=0
    )
    return loss_evaluator
