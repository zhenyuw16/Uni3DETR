# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import pyquaternion
import tempfile
from nuscenes.utils.data_classes import Box as NuScenesBox
from os import path as osp

from ..core.indoor_eval import indoor_eval_ov

import mmdet3d
#from mmdet.datasets import DATASETS
from mmdet3d.datasets import DATASETS
from mmdet3d.core import show_result
from mmdet3d.core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from mmdet3d.datasets import SUNRGBDDataset

__mmdet3d_version__ = float(mmdet3d.__version__[:3])

@DATASETS.register_module()
class SUNRGBDDataset_OV(SUNRGBDDataset):
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 seen_classes=None,
                 modality=dict(use_camera=True, use_lidar=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)

        self.seen_classes = seen_classes
        self.classes = seen_classes
    
    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 iou_thr_2d=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 axis_aligned_lw=False):
        """Evaluate.

        Evaluation in indoor protocol.

        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: None.
            iou_thr (list[float], optional): AP IoU thresholds for 3D
                evaluation. Default: (0.25, 0.5).
            iou_thr_2d (list[float], optional): AP IoU thresholds for 2D
                evaluation. Default: (0.5, ).
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict: Evaluation results.
        """
        assert isinstance(
            results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(
            results[0], dict
        ), f'Expect elements in results to be dict, got {type(results[0])}.'
        gt_annos = [info['annos'] for info in self.data_infos]
        label2cat = {i: cat_id for i, cat_id in enumerate(self.CLASSES)}
        ret_dict = indoor_eval_ov(
            self.seen_classes,
            gt_annos,
            results,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d,
            axis_aligned_lw=axis_aligned_lw)
        if show:
            self.show(results, out_dir, pipeline=pipeline)

        return ret_dict
