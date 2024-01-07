from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.runner import BaseModule
from mmdet.models import HEADS

from ..blocks import linear_relu_ln


@HEADS.register_module()
class MotionHead(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        fut_ts=12,
        fut_mode=6,
    ):
        self.embed_dims = embed_dims
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        self.mode_query = nn.Embedding(fut_mode, embed_dims)
        self.reg_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            Linear(embed_dims, fut_ts * 2),
        )
        self.cls_branch = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 2),
            Linear(self.embed_dims, 1),
        )

    def init_weights(self):
        nn.init.orthogonal_(self.mode_query.weight)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def forward(
        self, 
        det_result, 
        map_result, 
        det_features, 
        map_features
    ):
        instance_feature, anchor_embed = det_features
        map_instance_feature, map_anchor_embed = map_features
        bs, N = instance_feature.shape[:2]
        mode_query = self.mode_query.weight
        motion_query = (instance_feature + anchor_embed)[..., None, :] + mode_query[None, None]
        reg = self.reg_branch(motion_query).reshape(bs, N, self.fut_mode, self.fut_ts, 2)
        cls = self.cls_branch(motion_query).squeeze(-1)

        motion_output = dict(
            reg=reg,
            cls=cls,
        )

        return motion_output