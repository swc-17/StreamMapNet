import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import (BBOX_SAMPLERS, BBOX_ASSIGNERS)
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import (build_assigner, build_sampler)
from mmdet.core.bbox.assigners import (AssignResult, BaseAssigner)


@BBOX_SAMPLERS.register_module()
class SparsePoint3DTarget(object):
    def __init__(
        self,
        assigner=None,
    ):
        super(SparsePoint3DTarget, self).__init__()
        self.assigner = build_assigner(assigner)

    def sample(
        self,
        cls_preds,
        pts_preds,
        cls_targets,
        pts_targets,
    ):
        pts_targets  = [x.flatten(2, 3) if len(x.shape)==4 else x for x in pts_targets]
        indices = []
        for(cls_pred, pts_pred, cls_target, pts_target) in zip(
            cls_preds, pts_preds, cls_targets, pts_targets
        ):
            preds=dict(lines=pts_pred, scores=cls_pred)
            gts=dict(lines=pts_target, labels=cls_target)
            indice = self.assigner.assign(preds, gts)
            indices.append(indice)
        
        bs, num_pred, num_cls = cls_preds.shape
        output_cls_target = cls_targets[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        output_box_target = pts_preds.new_zeros(pts_preds.shape)
        output_reg_weights = pts_preds.new_zeros(pts_preds.shape)
        for i, (pred_idx, target_idx, gt_permute_index) in enumerate(indices):
            if len(cls_targets[i]) == 0:
                continue
            permute_idx = gt_permute_index[pred_idx, target_idx]
            output_cls_target[i, pred_idx] = cls_targets[i][target_idx]
            output_box_target[i, pred_idx] = pts_targets[i][target_idx, permute_idx]
            output_reg_weights[i, pred_idx] = 1

        return output_cls_target, output_box_target, output_reg_weights


@BBOX_ASSIGNERS.register_module()
class HungarianLinesAssigner_v2(BaseAssigner):
    """
        Computes one-to-one matching between predictions and ground truth.
        This class computes an assignment between the targets and the predictions
        based on the costs. The costs are weighted sum of three components:
        classification cost and regression L1 cost. The
        targets don't include the no_object, so generally there are more
        predictions than targets. After the one-to-one matching, the un-matched
        are treated as backgrounds. Thus each query prediction will be assigned
        with `0` or a positive integer indicating the ground truth index:
        - 0: negative sample, no assigned gt
        - positive integer: positive sample, index (1-based) of assigned gt
        Args:
            cls_weight (int | float, optional): The scale factor for classification
                cost. Default 1.0.
            bbox_weight (int | float, optional): The scale factor for regression
                L1 cost. Default 1.0.
    """

    def __init__(self, cost=dict, **kwargs):
        self.cost = build_match_cost(cost)

    def assign(self,
               preds: dict,
               gts: dict,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """
            Computes one-to-one matching based on the weighted costs.
            This method assign each query prediction to a ground truth or
            background. The `assigned_gt_inds` with -1 means don't care,
            0 means negative sample, and positive number is the index (1-based)
            of assigned gt.
            The assignment is done in the following steps, the order matters.
            1. assign every prediction to -1
            2. compute the weighted costs
            3. do Hungarian matching on CPU based on the costs
            4. assign all to 0 (background) first, then for each matched pair
            between predictions and gts, treat this prediction as foreground
            and assign the corresponding gt index (plus 1) to it.
            Args:
                lines_pred (Tensor): predicted normalized lines:
                    [num_query, num_points, 2]
                cls_pred (Tensor): Predicted classification logits, shape
                    [num_query, num_class].

                lines_gt (Tensor): Ground truth lines
                    [num_gt, num_points, 2].
                labels_gt (Tensor): Label of `gt_bboxes`, shape (num_gt,).
                gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                    labelled as `ignored`. Default None.
                eps (int | float, optional): A value added to the denominator for
                    numerical stability. Default 1e-7.
            Returns:
                :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        num_gts, num_lines = gts['lines'].size(0), preds['lines'].size(0)
        if num_gts == 0 or num_lines == 0:
            return None, None, None

        # compute the weighted costs
        gt_permute_idx = None # (num_preds, num_gts)
        if self.cost.reg_cost.permute:
            cost, gt_permute_idx = self.cost(preds, gts)
        else:
            cost = self.cost(preds, gts)

        # do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu().numpy()
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        return matched_row_inds, matched_col_inds, gt_permute_idx