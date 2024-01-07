# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models import build_head


@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
         self,
        task_config: dict,
        det_head = dict,
        map_head = dict,
        motion_head = dict,
        init_cfg=None,
        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        self.task_config = task_config
        if self.task_config['with_det']:
            self.det_head = build_head(det_head)
        if self.task_config['with_map']:
            self.map_head = build_head(map_head)
        if self.task_config['with_motion']:
            self.motion_head = build_head(motion_head)

    def init_weights(self):
        if self.task_config['with_det']:
            self.det_head.init_weights()
        if self.task_config['with_map']:
            self.map_head.init_weights()
        if self.task_config['with_motion']:
            self.motion_head.init_weights()

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        metas: dict,
        feature_queue=None,
        meta_queue=None,
    ):
        if self.task_config['with_det']:
            det_result, det_features = self.det_head(feature_maps, metas, feature_queue, meta_queue)
        else:
            det_result = None
 
        if self.task_config['with_map']:
            map_result, map_features = self.map_head(feature_maps, metas, feature_queue, meta_queue)
        else:
            map_result = None
        
        if self.task_config['with_motion']:
            motion_result = self.motion_head(det_result, map_result, det_features, map_features)
        else:
            motion_result = None
        
        return det_result, map_result, motion_result

    def loss(self, det_result, map_result, motion_result, data):
        losses = dict()
        if self.task_config['with_det']:
            cls_scores, reg_preds = det_result
            loss_agent = self.det_head.loss(cls_scores, reg_preds, data)
            losses.update(loss_agent)
        
        if self.task_config['with_map']:
            cls_scores, reg_preds = map_result
            loss_map = self.map_head.loss(cls_scores, reg_preds, data)
            losses.update(loss_map)

        return losses

    def post_process(self, det_result, map_result, motion_result):
        if self.task_config['with_det']:
            cls_scores, reg_preds = det_result
            det_output = self.det_head.post_process(cls_scores, reg_preds)
            batch_size = len(det_output)
        
        if self.task_config['with_map']:
            cls_scores, reg_preds = map_result
            map_output = self.map_head.post_process(cls_scores, reg_preds)
            batch_size = len(map_output)
        
        outputs = [dict()] * batch_size
        for i in range(batch_size):
            if self.task_config['with_det']:
                outputs[i].update(det_output[i])
            if self.task_config['with_map']:
                outputs[i].update(map_output[i])

        return outputs
