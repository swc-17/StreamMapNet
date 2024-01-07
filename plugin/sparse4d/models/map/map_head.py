# Copyright (c) Horizon Robotics. All rights reserved.
import torch

from mmcv.runner import force_fp32
from mmdet.models import HEADS
from mmdet.core import reduce_mean

from ..detection3d import Sparse4DDetHead


@HEADS.register_module()
class Sparse4DMapHead(Sparse4DDetHead):
    def __init__(
        self,
        roi_size=None,
        num_sample=20,
        **kwargs,
    ):
        super(Sparse4DMapHead, self).__init__(**kwargs)
        self.roi_size = roi_size
        self.num_sample = num_sample

    def normalize_line(self, line):
        bs, num_anchor, _ = line.shape
        line = line.view(bs, num_anchor, self.num_sample, -1)
        
        origin = -line.new_tensor([self.roi_size[0]/2, self.roi_size[1]/2])
        line = line - origin

        # transform from range [0, 1] to (0, 1)
        eps = 1e-5
        norm = line.new_tensor([self.roi_size[0], self.roi_size[1]]) + eps
        line = line / norm
        line = line.flatten(2, 3)

        return line

    @force_fp32(apply_to=("cls_scores", "reg_preds"))
    def loss(self, cls_scores, reg_preds, data, feature_maps=None):
        output = {}
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            reg = reg[..., : len(self.reg_weights)]
            reg = self.normalize_line(reg)
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
            mask_valid = mask.clone()

            num_pos = max(
                reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
            )
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                mask = torch.logical_and(
                    mask, cls.max(dim=-1).values.sigmoid() > threshold
                )
            
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            mask = mask.reshape(-1)
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
            reg_target = reg_target.flatten(end_dim=1)[mask]
            reg = reg.flatten(end_dim=1)[mask]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )
            reg_loss = self.loss_reg(
                reg, reg_target, weight=reg_weights, avg_factor=num_pos
            )

            output.update(
                {
                    f"map_loss_cls_{decoder_idx}": cls_loss,
                    f"map_loss_reg_{decoder_idx}": reg_loss,
                }
            )
        return output
