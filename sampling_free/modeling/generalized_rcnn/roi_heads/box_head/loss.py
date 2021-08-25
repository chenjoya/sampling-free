import torch
from torch.functional import F

from sampling_free.layers import smooth_l1_loss
from sampling_free.structures import boxlist_iou

from ...functions import BoxCoder, Matcher, BalancedPositiveNegativeSampler, OvSampler

class FastRCNNLossComputation(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.sampling_free = cfg.MODEL.SAMPLING_FREE
        if not self.sampling_free:
            self.fg_bg_sampler = BalancedPositiveNegativeSampler(
               cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
            ) if not cfg.MODEL.OVSAMPLER_ON else OvSampler(
                cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
            )
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
            cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
            allow_low_quality_matches=False,
        )
        self.box_coder = BoxCoder(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)
        self.offset = torch.tensor([0, 1, 2, 3], device="cuda")

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs, matched_vals = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields("labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        matched_targets.add_field("matched_vals", matched_vals)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels, max_eg_ious = [], []
        regression_targets = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            
            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            max_eg_ious.append(matched_targets.get_field("matched_vals"))
            regression_targets.append(regression_targets_per_image)
        
        return labels, max_eg_ious, regression_targets

    def subsample(self, proposals, targets):
        labels, max_eg_ious, regression_targets = self.prepare_targets(proposals, targets)
        proposals = list(proposals)

        # add corresponding label and regression_targets information to the bounding boxes
        for img_idx, (labels_per_image, regression_targets_per_image, 
            proposals_per_image) in enumerate(zip(labels, regression_targets, proposals)):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("regression_targets", 
                regression_targets_per_image)
            if self.sampling_free:
                proposals[img_idx] = proposals[img_idx][labels_per_image >= 0]
        
        if not self.sampling_free:
            sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels, 
                proposals, max_eg_ious)
            # distributed sampled proposals, that were obtained on all feature maps
            # concatenated via the fg_bg_sampler, into individual feature map levels
            for img_idx, (pos_inds_img, neg_inds_img) in enumerate(
                zip(sampled_pos_inds, sampled_neg_inds)
            ):
                img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
                proposals_per_image = proposals[img_idx][img_sampled_inds]
                proposals[img_idx] = proposals_per_image

        self._proposals = proposals
        return proposals

    def __call__(self, class_logits, box_regression):
        if not hasattr(self, "_proposals"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposals

        labels = torch.cat([proposal.get_field("labels") for proposal in proposals], dim=0)
        regression_targets = torch.cat(
            [proposal.get_field("regression_targets") for proposal in proposals], dim=0
        )

        sampled_pos_inds_subset = torch.nonzero(labels > 0).squeeze(1)
        labels_pos = labels[sampled_pos_inds_subset]
        map_inds = 4 * labels_pos[:, None] + self.offset

        valid = labels >= 0
        box_cls_loss = F.cross_entropy(class_logits[valid], labels[valid])
        box_loc_loss = smooth_l1_loss(
            box_regression[sampled_pos_inds_subset[:, None], map_inds],
            regression_targets[sampled_pos_inds_subset],
            size_average=False,
            beta=1,
        )
        
        if self.sampling_free:
            box_loc_loss = box_loc_loss / (labels_pos.numel() * 4)
            with torch.no_grad():
                ratio = box_loc_loss / box_cls_loss
            box_cls_loss = 2 * ratio * box_cls_loss
        else:
            box_loc_loss = box_loc_loss / labels.numel()
        return dict(box_cls_loss=box_cls_loss, box_loc_loss=box_loc_loss)

def make_roi_box_loss_evaluator(cfg):
    return FastRCNNLossComputation(cfg)
