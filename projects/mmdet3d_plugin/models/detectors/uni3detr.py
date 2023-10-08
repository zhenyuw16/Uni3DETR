from re import I
from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from mmcv.cnn import Conv2d
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet.core import multi_apply
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.core.merge_all_augs import merge_all_aug_bboxes_3d

from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import gather_points
import torch.nn.functional as F

def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
        ]

    if pred_xyz.ndim == 4:
        src_range = [x[:, None] for x in src_range]
        dst_range = [x[:, None] for x in dst_range]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
    dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
    prop_xyz = (
        ((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff
    ) + dst_range[0][:, None, :]
    return prop_xyz



class PointSample(object):
    """Point sample.
    Sampling data to a certain number.
    Args:
        num_points (int): Number of points to be sampled.
        sample_range (float, optional): The range where to sample points.
            If not None, the points with depth larger than `sample_range` are
            prior to be sampled. Defaults to None.
        replace (bool, optional): Whether the sampling is with or without
            replacement. Defaults to False.
    """

    def __init__(self, num_points, sample_range=None, replace=False):
        self.num_points = num_points
        self.sample_range = sample_range
        self.replace = replace

    def _points_random_sampling(self,
                                points,
                                num_samples,
                                sample_range=None,
                                replace=False,
                                return_choices=False):
        """Points random sampling.
        Sample points to a certain number.
        Args:
            points (np.ndarray | :obj:`BasePoints`): 3D Points.
            num_samples (int): Number of samples to be sampled.
            sample_range (float, optional): Indicating the range where the
                points will be sampled. Defaults to None.
            replace (bool, optional): Sampling with or without replacement.
                Defaults to None.
            return_choices (bool, optional): Whether return choice.
                Defaults to False.
        Returns:
            tuple[np.ndarray] | np.ndarray:
                - points (np.ndarray | :obj:`BasePoints`): 3D Points.
                - choices (np.ndarray, optional): The generated random samples.
        """
        if not replace:
            replace = (points.shape[0] < num_samples)
        point_range = range(len(points))
        if sample_range is not None and not replace:
            # Only sampling the near points when len(points) >= num_samples
            dist = np.linalg.norm(points.tensor, axis=1)
            far_inds = np.where(dist >= sample_range)[0]
            near_inds = np.where(dist < sample_range)[0]
            # in case there are too many far points
            if len(far_inds) > num_samples:
                far_inds = np.random.choice(
                    far_inds, num_samples, replace=False)
            point_range = near_inds
            num_samples -= len(far_inds)
        choices = np.random.choice(point_range, num_samples, replace=replace)
        if sample_range is not None and not replace:
            choices = np.concatenate((far_inds, choices))
            # Shuffle points after sampling
            np.random.shuffle(choices)
        if return_choices:
            return points[choices], choices
        else:
            return points[choices]

@DETECTORS.register_module()
class Uni3DETR(MVXTwoStageDetector):
    """Uni3DETR."""
    def __init__(self,
                 dynamic_voxelization=False,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None, pretrained=None,):
        super(Uni3DETR, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             None, pts_backbone, None, pts_neck, pts_bbox_head, None, None, 
                             train_cfg, test_cfg, pretrained)
        
        self.dynamic_voxelization = dynamic_voxelization

        if pts_middle_encoder:
            self.pts_fp16 = True if hasattr(self.pts_middle_encoder, 'fp16_enabled') else False
        
        self.fps_module = Points_Sampler([pts_bbox_head['num_query']]) 

    def init_weights(self):
        return

    @force_fp32()
    def extract_pts_feat(self, pts):
        """Extract features of points."""

        if not self.dynamic_voxelization:
            voxels, num_points, coors = self.voxelize(pts)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            if not self.pts_fp16:
                voxel_features = voxel_features.float()
            
            batch_size = coors[-1, 0] + 1
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        else:
            coors = []
            for res in pts:
                res_coors = self.pts_voxel_layer(res)
                coors.append(res_coors)
            points = torch.cat(pts, dim=0)
            coors_batch = []
            for i, coor in enumerate(coors):
                coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)
            voxels, coors = points, coors_batch
            voxel_features, feature_coors = self.pts_voxel_encoder(voxels, coors)
            if not self.pts_fp16:
                voxel_features = voxel_features.float()
            batch_size = coors[-1, 0] + 1
            x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)

        bpts = [i.unsqueeze(0).float() for i in pts]
        ind = [self.fps_module(i, None) for i in bpts]
        fpsbpts = torch.cat([gather_points(bpts[i][:, :, :3].transpose(1,2).contiguous(), ind[i]).transpose(1,2).contiguous() for i in range(batch_size)] )
        fpsbpts = shift_scale_points(fpsbpts, src_range=[fpsbpts.min(dim=1)[0], fpsbpts.max(dim=1)[0] ] )

        bpts = [coors[coors[:,0]==i, 1:].unsqueeze(0).float() for i in range(batch_size)]
        ind = [self.fps_module(i, None) for i in bpts]
        fpsbpts2 = torch.cat([gather_points(bpts[i][:, :, :3].transpose(1,2).contiguous(), ind[i]).transpose(1,2).contiguous() for i in range(batch_size)] )
        fpsbpts2 = fpsbpts2[:, :, [2,1,0]].float()
        fpsbpts2 = shift_scale_points(fpsbpts2, src_range=[fpsbpts2.min(dim=1)[0], fpsbpts2.max(dim=1)[0] ] )

        fpsbpts = torch.cat([fpsbpts, fpsbpts2], 1)
        return x, fpsbpts

    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None, 
                          fpsbpts=None):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_metas, fpsbpts)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    @force_fp32(apply_to=('points'))
    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        pts_feat, fpsbpts = self.extract_pts_feat(points)

        losses = dict()
        losses_pts = self.forward_pts_train(pts_feat, gt_bboxes_3d, gt_labels_3d, 
                                            img_metas, gt_bboxes_ignore, fpsbpts)
        losses.update(losses_pts)

        return losses
    
    def forward_test(self, img_metas, points=None, **kwargs):
        #img_metas = img_metas.data
        #points = points.data[0]
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_metas)
        if points is not None:
            if num_augs != len(points):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(points), len(img_metas)))

        if num_augs == 1:
            if not isinstance(img_metas[0], list):
                img_metas = [img_metas]
            results = self.simple_test(img_metas[0], points, **kwargs)
        else:
            results = self.aug_test(points, img_metas, **kwargs)

        return results

    def simple_test_pts(self, pts_feat, img_metas, rescale=False, fpsbpts=None):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(pts_feat, img_metas, fpsbpts)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, points=None, rescale=False):
        """Test function without augmentaiton."""
        pts_feat, fpsbpts = self.extract_pts_feat(points)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feat, img_metas, rescale=rescale, fpsbpts=fpsbpts)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            # result_dict['pts_bbox'] = pts_bbox
            # result_dict = pts_bbox
            for k in pts_bbox.keys():
                result_dict[k] = pts_bbox[k]
        
        return bbox_list


    ################ not done #####################
    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        pts_feats, img_feats, img_depths, fpsbpts = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_feats, img_depths, img_metas, rescale, fpsbpts)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        if points is None:
            points = [None] * len(img_metas)
        pts_feats, img_feats, img_depths, fpsbpts= multi_apply(self.extract_feat, points, imgs,
                                                       img_metas)
        return pts_feats, img_feats, img_depths, fpsbpts

    def aug_test_pts(self, pts_feats, img_feats, img_depths, img_metas, rescale=False, fpsbpts=None):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for _idx, img_meta in enumerate(img_metas):
            outs = self.pts_bbox_head(pts_feats[_idx], img_feats[_idx], 
                                      img_meta, img_depths[_idx], fpsbpts[_idx])
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_meta, rescale=rescale)

            bbox_list = [
                bbox3d2result(bboxes, scores, labels, ious)
                    for bboxes, scores, labels, ious in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_all_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes
