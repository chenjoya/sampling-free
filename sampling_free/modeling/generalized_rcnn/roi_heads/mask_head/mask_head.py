import torch

from sampling_free.structures import BoxList

from .roi_mask_feature_extractors import make_roi_mask_feature_extractor
from .roi_mask_predictors import make_roi_mask_predictor
from .inference import make_roi_mask_post_processor
from .loss import make_roi_mask_loss_evaluator

def keep_only_positive_boxes(boxes):
    assert isinstance(boxes, (list, tuple))
    assert isinstance(boxes[0], BoxList)
    assert boxes[0].has_field("labels")
    positive_boxes = []
    for boxes_per_image in boxes:
        inds_mask = boxes_per_image.get_field("labels") > 0
        inds = inds_mask.nonzero().squeeze(1)
        positive_boxes.append(boxes_per_image[inds])
    return positive_boxes

class ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIMaskHead, self).__init__()
        self.feature_extractor = make_roi_mask_feature_extractor(
            cfg, in_channels)
        self.predictor = make_roi_mask_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        if self.training:
            proposals = keep_only_positive_boxes(proposals)
        x = self.feature_extractor(features, proposals)
        mask_logits = self.predictor(x)
        if self.training:
            return self.loss_evaluator(proposals, mask_logits, targets)
        else:
            return self.post_processor(mask_logits, proposals)

def build_roi_mask_head(cfg, in_channels):
    return ROIMaskHead(cfg, in_channels)
