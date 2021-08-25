import torch

from .balanced_positive_negative_sampler import BalancedPositiveNegativeSampler
from .box_coder import BoxCoder
from .make_layers import group_norm, conv_with_kaiming_uniform, make_conv3x3, make_fc
from .matcher import Matcher
from .poolers import Pooler
from .registry import *
from .guided_loss import GuidedLoss
from .ovsampler import OvSampler

def cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)