import torch

from .box_head import build_roi_box_head
from .mask_head import build_roi_mask_head

class CombinedROIHeads(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super().__init__()
        self.box = build_roi_box_head(cfg, in_channels)
        if cfg.MODEL.MASK_ON:
            self.mask = build_roi_mask_head(cfg, in_channels)

    def forward_train(self, features, proposals, targets=None):
        proposals, losses = self.box(features, proposals, targets)
        if hasattr(self, "mask"):
            losses.update(self.mask(features, proposals, targets))
        return losses
    
    def forward_test(self, features, proposals):
        results = self.box(features, proposals)
        if hasattr(self, "mask"):
            results = self.mask(features, results)
        return results
    
    def forward(self, features, proposals, targets=None):
        if self.training:
            return self.forward_train(features, proposals, targets)
        else:
            return self.forward_test(features, proposals)

def build_roi_heads(cfg, in_channels):
    return CombinedROIHeads(cfg, in_channels)
