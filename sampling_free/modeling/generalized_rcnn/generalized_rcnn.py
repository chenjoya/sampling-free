import torch
from torch import nn

from sampling_free.structures import to_image_list
from .backbone import build_backbone
from .rpn import build_rpn
from .roi_heads import build_roi_heads

class GeneralizedRCNN(nn.Module):
    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        if cfg.MODEL.RCNN_ON:
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.device = torch.device("cuda")

    def forward(self, batches):
        if self.training:
            images, targets, _ = batches 
            return self.forward_train(images, targets)
        else:
            return self.forward_test(batches)
    
    def forward_train(self, images, targets):
        images = to_image_list(images).to(self.device)
        targets = [target.to(self.device) for target in targets]
        features = self.backbone(images.tensors)
        proposals, losses = self.rpn(images, features, targets)
        if hasattr(self, "roi_heads"):
            losses.update(self.roi_heads(features, proposals, targets))
        return losses
    
    def forward_test(self, images):
        images = to_image_list(images).to(self.device)
        features = self.backbone(images.tensors)
        results = self.rpn(images, features)
        if hasattr(self, "roi_heads"):
            results = self.roi_heads(features, results)
        return results