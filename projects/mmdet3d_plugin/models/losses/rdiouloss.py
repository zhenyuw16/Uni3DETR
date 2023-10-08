# Copyright (c) OpenMMLab. All rights reserved.
import torch
from projects.mmdet3d_plugin.core.bbox.util import get_rdiou
from torch import nn as nn
import torch.nn.functional as F

from mmdet.models.losses.utils import weighted_loss
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models import LOSSES
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d, bbox_overlaps_nearest_3d

@weighted_loss
def rd_iou_loss(pred, target):
    """Calculate the IoU loss (1-IoU) of two sets of rotated bounding boxes.
    Note that predictions and targets are one-to-one corresponded.

    Args:
        pred (torch.Tensor): Bbox predictions with shape [N, 7]
            (x, y, z, w, l, h, alpha).
        target (torch.Tensor): Bbox targets (gt) with shape [N, 7]
            (x, y, z, w, l, h, alpha).

    Returns:
        torch.Tensor: IoU loss between predictions and targets.
    """
    u, rdiou = get_rdiou(pred.unsqueeze(0), target.unsqueeze(0))
    u, rdiou = u[0], rdiou[0]

    rdiou_loss_n = rdiou - u
    rdiou_loss_n = torch.clamp(rdiou_loss_n,min=-1.0,max = 1.0)
    rdiou_loss_n = 1 - rdiou_loss_n
    return rdiou_loss_n


@LOSSES.register_module()
class RDIoULoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of rotated bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            weight (torch.Tensor | float, optional): Weight of loss.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        loss = self.loss_weight * rd_iou_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss


@weighted_loss
def iou3d_loss(pred, target):
    #iou3d = bbox_overlaps_3d(pred, target, coordinate='depth')
    #iou3d = 1 - torch.diag(iou3d)

    #iou3d = (1 - bbox_overlaps_nearest_3d(pred, target, is_aligned=True, coordinate='depth') )
    iou3d = (1 - bbox_overlaps_nearest_3d(pred, target, is_aligned=True, coordinate='lidar') ) 
    #iou3d += (1 - bbox_overlaps_nearest_3d(pred[:, [0,2,1,3,5,4,6]], target[:, [0,2,1,3,5,4,6]], is_aligned=True, coordinate='depth') )  * 0.1
    #iou3d += (1 - bbox_overlaps_nearest_3d(pred[:, [1,2,0,4,5,3,6]], target[:, [1,2,0,4,5,3,6]], is_aligned=True, coordinate='depth') ) * 0.1
    return iou3d


@LOSSES.register_module()
class IoU3DLoss(nn.Module):
    """Calculate the IoU loss (1-IoU) of rotated bounding boxes.

    Args:
        reduction (str): Method to reduce losses.
            The valid reduction method are none, sum or mean.
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss calculation.

        Args:
            pred (torch.Tensor): Bbox predictions with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            target (torch.Tensor): Bbox targets (gt) with shape [..., 7]
                (x, y, z, w, l, h, alpha).
            weight (torch.Tensor | float, optional): Weight of loss.
                Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): Method to reduce losses.
                The valid reduction method are 'none', 'sum' or 'mean'.
                Defaults to None.

        Returns:
            torch.Tensor: IoU loss between predictions and targets.
        """
        if weight is not None and not torch.any(weight > 0):
            return pred.sum() * weight.sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)
        loss = self.loss_weight * iou3d_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        return loss

def soft_focal_loss(pred,
        target,
        weight=None,
        gamma=2.0,
        alpha=0.25,
        reduction='mean',
        avg_factor=None):
    pred_sigmoid = pred.sigmoid()

    target, target_score = target[0], target[1]
    target_oh = torch.zeros((pred_sigmoid.shape[0], pred.shape[1] + 1)).type_as(pred).to(pred.device)
    target_oh.scatter_(1, target[:,None], 1)
    target_oh = target_oh[:,0:-1]
    target = target[:,None]

    target_soft = (target_oh > 0).float() * target_score[:,None]
    pt = target_soft - pred_sigmoid
    focal_weight = ((1 - alpha) + (2*alpha - 1) * target_soft) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target_soft, reduction='none') * focal_weight

    weight = weight.view(-1,1)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

@LOSSES.register_module()
class SoftFocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.25,
                 reduction='mean',
                 loss_weight=1.0):
        super(SoftFocalLoss, self).__init__()
        assert use_sigmoid is True, 'Only sigmoid focal loss supported now.'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight
    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            loss_cls = self.loss_weight * soft_focal_loss(
                pred,
                target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=reduction,
                avg_factor=avg_factor)
        else:
            raise NotImplementedError
        return loss_cls