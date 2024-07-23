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
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask

from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import gather_points
import torch.nn.functional as F

from ..utils import Uni3DViewTrans
from .uni3detr import shift_scale_points, PointSample


@DETECTORS.register_module()
class OV_Uni3DETR(MVXTwoStageDetector):
    """OV_Uni3DETR."""
    def __init__(self,
                 dynamic_voxelization=False,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 depth_head=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None, 
                 pretrained=None,
                 view_cfg=None,
                 multimodal=False,
                 pretrained_img=None,
                 pretrained_pts=None,
                 load_img=None,
                 load_pts=None):
        super(OV_Uni3DETR, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck, pts_bbox_head, img_roi_head, img_rpn_head, 
                             train_cfg, test_cfg, pretrained)
        
        self.dynamic_voxelization = dynamic_voxelization

        if pts_middle_encoder:
            self.pts_fp16 = True if hasattr(self.pts_middle_encoder, 'fp16_enabled') else False
        
        self.fps_module = Points_Sampler([pts_bbox_head['num_query']]) 
        
        if img_backbone is not None:
            self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
            in_channels = self.img_neck.out_channels
            out_channels = self.pts_bbox_head.in_channels
            if isinstance(in_channels, list):
                in_channels  = in_channels[0]
            self.input_proj = Conv2d(in_channels, out_channels, kernel_size=1)
            if "SimpleDepth" in depth_head.type:
                self.depth_dim = depth_head.model.depth_dim
                self.depth_net = Conv2d(out_channels, self.depth_dim, kernel_size=1)
            else:
                raise NotImplementedError
            self.depth_head = depth_head
            self.use_grid_mask = use_grid_mask
        
        if view_cfg is not None:
            self.view_trans = Uni3DViewTrans(**view_cfg)\
        
        self.pretrained_img = pretrained_img
        self.pretrained_pts = pretrained_pts
        self.load_img = load_img
        self.load_pts = load_pts
        
        if multimodal:
            self.pts_conv = nn.Sequential(
                            nn.Conv3d(view_cfg['embed_dims'], ############ *2
                                    view_cfg['embed_dims'],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True),
                            nn.BatchNorm3d(view_cfg['embed_dims']),
                            nn.ReLU(inplace=True))
            self.img_conv = nn.Sequential(
                            nn.Conv3d(view_cfg['embed_dims'], ############ *2
                                    view_cfg['embed_dims'],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True),
                            nn.BatchNorm3d(view_cfg['embed_dims']),
                            nn.ReLU(inplace=True))
            self.conv_layer = []
            for k in range(1):
                conv = nn.Sequential(
                            nn.Conv3d(view_cfg['embed_dims'] *2, ############ *2
                                    view_cfg['embed_dims'],
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    bias=True),
                            nn.BatchNorm3d(view_cfg['embed_dims']),
                            nn.ReLU(inplace=True))
                self.add_module("{}_head_{}".format('conv_trans', k + 1), conv)
                self.conv_layer.append(conv)

    def init_weights(self):
        # load pretrained pts model
        if self.pretrained_pts is not None:
            ckpt_load = torch.load(self.pretrained_pts, 
                           map_location="cuda:{}".format(torch.cuda.current_device()))["state_dict"]
            print("Loaded pretrained model from: {}".format(self.pretrained_pts))
            for load_key in self.load_pts:
                dict_load = {_key.replace(load_key+'.',''):ckpt_load[_key] 
                            for _key in ckpt_load if load_key in _key}
                getattr(self, load_key).load_state_dict(dict_load, strict=False)
                print("Loaded pretrained {}".format(load_key))
                assert len(dict_load) > 0

        # load pretrained img model
        if self.pretrained_img is not None:
            ckpt_load = torch.load(self.pretrained_img, 
                           map_location="cuda:{}".format(torch.cuda.current_device()))["state_dict"]
            print("Loaded pretrained img model from: {}".format(self.pretrained_img))

            # load pts key for teacher
            for load_key in self.load_img:
                if 'img' not in load_key: continue
                dict_load = {_key.replace(load_key+'.',''):ckpt_load[_key] 
                        for _key in ckpt_load if load_key in _key}
                getattr(self, load_key).load_state_dict(dict_load, strict=False)
                print("Loaded pretrained {}".format(load_key))
                assert len(dict_load) > 0

            if 'input_proj' in self.load_img:
                dict_load = {_key.replace('input_proj.',''):ckpt_load[_key] 
                            for _key in ckpt_load if 'input_proj' in _key}
                self.input_proj.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained input_proj")
                assert len(dict_load) > 0

            if 'depth_head' in self.load_img:
                dict_load = {_key.replace('depth_net.',''):ckpt_load[_key] 
                            for _key in ckpt_load if 'depth_net' in _key}
                self.depth_net.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained depth_net")
                assert len(dict_load) > 0

            if 'view_trans' in self.load_img:
                dict_load = {_key.replace('view_trans.',''):ckpt_load[_key] 
                            for _key in ckpt_load if 'view_trans' in _key}
                self.view_trans.load_state_dict(dict_load, strict=False)
                print("Loaded pretrained view_trans")
                assert len(dict_load) > 0
    
    @property
    def with_depth_head(self):
        """bool: Whether the detector has a depth head."""
        return hasattr(self, 'depth_head') and self.depth_head is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if img is not None:
            B = img.size(0)
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if False: #img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5: # and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            # print(img.shape)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None        
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = self.input_proj(img_feat)
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
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


    @auto_fp16(apply_to=('img'))
    def pred_depth(self, img, img_metas, img_feats=None):
        if img_feats is None:
            return None
        
        B = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.view(B * N, C, H, W)
        if self.depth_head.type == "SimpleDepth":
            depth = []
            for _feat in img_feats:
                _depth = self.depth_net(_feat.view(-1,*_feat.shape[-3:]))
                _depth = _depth.softmax(dim=1)
                depth.append(_depth)
        else:
            raise NotImplementedError
        return depth
    
    def extract_feats(self, points=None,
                      img_metas=None,
                      img=None):
        if points is not None and img is not None:
            pts_feats, fpsbpts = self.extract_pts_feat(points)
            img_feats = self.extract_img_feat(img, img_metas)
            img_depth = self.pred_depth(img=img, img_metas=img_metas, img_feats=img_feats)
            img_feats = self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)

            if len(pts_feats.shape) == 5:
                pts_feats = pts_feats.unsqueeze(1)

            raw_shape = img_feats.shape

            if pts_feats.requires_grad:
                ri = np.random.randint(0,3)
                if ri == 0:
                    unified_feats = torch.cat([img_feats.flatten(1,2), img_feats.flatten(1,2)], 1)
                elif ri == 1:
                    unified_feats = torch.cat([pts_feats.flatten(1,2), pts_feats.flatten(1,2)], 1)
                else:
                    unified_feats = torch.cat([pts_feats.flatten(1,2), img_feats.flatten(1,2)], 1)
            else:
                unified_feats = torch.cat([pts_feats.flatten(1,2), img_feats.flatten(1,2)], 1)
                # unified_feats = torch.cat([img_feats.flatten(1,2), img_feats.flatten(1,2)], 1)
                
            
            for layer in self.conv_layer:
                unified_feats = layer(unified_feats)

            pts_feat = unified_feats.reshape(*raw_shape)
        elif points is not None:
            pts_feat, fpsbpts = self.extract_pts_feat(points)
        elif img is not None:
            img_feats = self.extract_img_feat(img, img_metas)
            img_depth = self.pred_depth(img=img, img_metas=img_metas, img_feats=img_feats)
            img_feats = self.view_trans(img_feats, img_metas=img_metas, img_depth=img_depth)
            pts_feat, fpsbpts = img_feats, None
        
        return pts_feat, fpsbpts


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

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
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
        pts_feat, fpsbpts = self.extract_feats(points, img_metas, img)

        losses = dict()
        losses_pts = self.forward_pts_train(pts_feat, gt_bboxes_3d, gt_labels_3d, 
                                            img_metas, gt_bboxes_ignore, fpsbpts)
        losses.update(losses_pts)

        return losses
    
    def forward_test(self, img_metas, points=None, img=None, **kwargs):
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
            if not isinstance(img, list):
                img = [img]
            results = self.simple_test(img_metas[0], points, img[0], **kwargs)
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
    
    def simple_test(self, img_metas, points=None, img=None, rescale=False):
        """Test function without augmentaiton."""
        # pts_feat, fpsbpts = self.extract_pts_feat(points)
        pts_feat, fpsbpts = self.extract_feats(points, img_metas, img)
        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            pts_feat, img_metas, rescale=rescale, fpsbpts=fpsbpts)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            # result_dict['pts_bbox'] = pts_bbox
            # result_dict = pts_bbox
            for k in pts_bbox.keys():
                result_dict[k] = pts_bbox[k]
        
        return bbox_list


    