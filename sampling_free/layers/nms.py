# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
from sampling_free import _C

nms = _C.nms
ml_nms = _C.ml_nms
# nms.__doc__ = """
# This function performs Non-maximum suppresion"""