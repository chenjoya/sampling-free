import torch

from .roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_box_predictors import make_roi_box_predictor
from .inference import make_roi_box_post_processor
from .loss import make_roi_box_loss_evaluator

class ROIBoxHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(ROIBoxHead, self).__init__()
        self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
        self.predictor = make_roi_box_predictor(
            cfg, self.feature_extractor.out_channels)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)
    
    def forward(self, x, proposals, targets=None):
        if self.training:
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)
        x = self.feature_extractor(x, proposals)
        class_logits, box_regression = self.predictor(x)
        if self.training:
            return proposals, self.loss_evaluator(class_logits, box_regression)
        return self.post_processor((class_logits, box_regression), proposals)

def build_roi_box_head(cfg, in_channels):
    return ROIBoxHead(cfg, in_channels)
