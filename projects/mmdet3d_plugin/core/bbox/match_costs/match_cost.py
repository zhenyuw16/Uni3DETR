import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
from mmcv.ops import diff_iou_rotated_3d
from mmdet3d.core.bbox import AxisAlignedBboxOverlaps3D
from projects.mmdet3d_plugin.core.bbox.util import get_rdiou
from mmdet3d.core.bbox.iou_calculators.iou3d_calculator import bbox_overlaps_3d, bbox_overlaps_nearest_3d
import torch.nn.functional as F

@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class RotatedIoU3DCost(object):

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        #print(bbox_pred.shape, gt_bboxes.shape)
        N = gt_bboxes.shape[0]
        M = bbox_pred.shape[0]
        bbox_costs = [diff_iou_rotated_3d(bbox_pred.unsqueeze(0), gt_bboxes[[i], :].repeat(M, 1).unsqueeze(0))[0].unsqueeze(1) for i in range(N)]
        bbox_cost = torch.cat(bbox_costs, 1)

        return bbox_cost * self.weight


@MATCH_COST.register_module()
class AxisAlignedIoU3DCost(object):

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        axis_aligned_iou = AxisAlignedBboxOverlaps3D()(bbox_pred, gt_bboxes)
        iou_loss = - axis_aligned_iou
        return iou_loss * self.weight

@MATCH_COST.register_module()
class RDIoUCost(object):

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        u, rdiou = get_rdiou(bbox_pred.unsqueeze(1), gt_bboxes.unsqueeze(0))

        rdiou_loss_n = rdiou - u
        rdiou_loss_n = torch.clamp(rdiou_loss_n,min=-1.0,max = 1.0)
        rdiou_loss_n = 1 - rdiou_loss_n
        return rdiou_loss_n * self.weight

@MATCH_COST.register_module()
class IoU3DCost(object):

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        #iou3d = 1 - bbox_overlaps_3d(bbox_pred, gt_bboxes, coordinate='depth')
        #iou3d = (1 - bbox_overlaps_nearest_3d(bbox_pred, gt_bboxes, coordinate='depth') ) 
        iou3d = (1 - bbox_overlaps_nearest_3d(bbox_pred, gt_bboxes, coordinate='lidar') ) ############
        #iou3d += (1 - bbox_overlaps_nearest_3d(bbox_pred[:, [0,2,1,3,5,4,6]], gt_bboxes[:, [0,2,1,3,5,4,6]], coordinate='depth') ) * 0.1
        #iou3d += (1 - bbox_overlaps_nearest_3d(bbox_pred[:, [1,2,0,4,5,3,6]], gt_bboxes[:, [1,2,0,4,5,3,6]], coordinate='depth') ) * 0.1
        return iou3d * self.weight


@MATCH_COST.register_module()
class SoftFocalLossCost(object):

    def __init__(self,
                 weight=1.,
                 alpha=0.25,
                 gamma=2,
                 eps=1e-12,
                 binary_input=False):
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.binary_input = binary_input


    def __call__(self, cls_pred, gt_labels, iou3d):

        cls_pred = cls_pred.sigmoid()

        iou3d = iou3d.pow(0.001)
        neg_cost = -(1 - cls_pred * iou3d + self.eps).log() * (
            1 - self.alpha) * (cls_pred * iou3d).pow(self.gamma)

        pos_cost = -(cls_pred * iou3d + self.eps).log() * self.alpha * (
            1 - cls_pred * iou3d).pow(self.gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]

        return cls_cost * self.weight