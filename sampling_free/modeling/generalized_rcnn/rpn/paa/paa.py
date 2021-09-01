import math

import torch
from torch import nn

from sampling_free.layers import Scale, DFConv2d

from ..anchor_generator import make_anchor_generator_paa
from ..atss.atss import BoxCoder
from .inference import make_paa_postprocessor
from .loss import make_paa_loss_evaluator

class PAAHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(PAAHead, self).__init__()
        num_classes = cfg.MODEL.PAA.NUM_CLASSES - 1
        num_anchors = len(cfg.MODEL.PAA.ASPECT_RATIOS) * cfg.MODEL.PAA.SCALES_PER_OCTAVE
        
        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.PAA.NUM_CONVS):
            if cfg.MODEL.PAA.USE_DCN_IN_TOWER and \
                    i == cfg.MODEL.PAA.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_anchors * num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, num_anchors * 4, kernel_size=3, stride=1,
            padding=1
        )
        self.iou_pred = nn.Conv2d(
            in_channels, num_anchors * 1, kernel_size=3, stride=1,
            padding=1
        )
        all_modules = [self.cls_tower, self.bbox_tower,
            self.cls_logits, self.bbox_pred, self.iou_pred
        ]
        
        # initialization
        for modules in all_modules:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.PAA.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits, bbox_reg, iou_pred = [], [], []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_reg.append(bbox_pred)
            iou_pred.append(self.iou_pred(box_tower))
        return [logits, bbox_reg, iou_pred] 

class PAAModule(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        super(PAAModule, self).__init__()
        self.head = PAAHead(cfg, in_channels)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_paa_loss_evaluator(cfg, box_coder)
        self.box_selector_test = make_paa_postprocessor(cfg, box_coder)
        self.anchor_generator = make_anchor_generator_paa(cfg)
        self.fpn_strides = cfg.MODEL.PAA.ANCHOR_STRIDES

    def forward(self, images, features, targets=None):
        preds = self.head(features)
        box_cls, box_regression, iou_pred = preds
        anchors = self.anchor_generator(images, features)
        locations = self.compute_locations(features)

        if self.training:
            return None, self.loss_evaluator(
                box_cls, box_regression, iou_pred, targets, anchors, locations
            )
        else:
            return self.box_selector_test(box_cls, box_regression, iou_pred, anchors)

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_paa(cfg, in_channels):
    return PAAModule(cfg, in_channels)
