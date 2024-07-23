_base_ = [
    '../../../configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
voxel_size = [0.02, 0.02, 0.02]
grid_size = [128, 320, 320]
point_cloud_range = [-3.2, -0.2, -2., 3.2, 6.2, 0.56]


cam_sweep_num = 1
fp16_enabled = False
bev_stride = 8 
sample_num = 15
voxel_shape = [int(((point_cloud_range[3]-point_cloud_range[0])/voxel_size[0])//bev_stride),
               int(((point_cloud_range[4]-point_cloud_range[1])/voxel_size[1])//bev_stride),
               sample_num]


input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    cam_sweep_num=cam_sweep_num)

model = dict(
    type='OV_Uni3DETR',
    use_grid_mask=True,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
        ),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
        ),
    depth_head=dict(
        type='SimpleDepth',
        model=dict(
            depth_dim=64,
        )),
    view_cfg=dict(
        num_cams=1,
        num_convs=3,
        num_points=sample_num,
        num_sweeps=cam_sweep_num,
        kernel_size=(3,3,3),
        keep_sweep_dim=True,
        num_feature_levels=4,
        embed_dims=256,
        pc_range=point_cloud_range,
        voxel_shape=voxel_shape,
        fp16_enabled=fp16_enabled,
    ),
    pts_bbox_head=dict(
        type='Uni3DETRHeadCLIP',
        num_query=300,
        zeroshot_path='clip_embed/sunrgbd_clip_a+cname_rn50_manyprompt_46c_coda.npy',
        num_classes=46, 
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=8,
        transformer=dict(
            type='Uni3DETRTransformer',
            fp16_enabled=fp16_enabled,
            decoder=dict(
                type='Uni3DETRTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='UniCrossAtten',
                            num_points=1,
                            embed_dims=256,
                            num_sweeps=cam_sweep_num,
                            fp16_enabled=fp16_enabled)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    norm_cfg=dict(type='LN'),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=1000,
            voxel_size=voxel_size,
            alpha=1.0,
            num_classes=46
            ), 
        post_processing=dict(
            type='nms',
            nms_thr=0.5),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(type='SoftFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='IoU3DLoss', loss_weight=1.2),
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=bev_stride,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoU3DCost', weight=1.2), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd_coda/'

# img_norm_cfg = dict(
#     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

class_names = ('chair', 'table', 'pillow', 'sofa_chair', 'desk', 'bed', 'sofa', 'computer', 'box', 
              'lamp', 'garbage_bin', 'cabinet', 'shelf', 'drawer', 'sink', 'night_stand', 'kitchen_counter', 
              'paper', 'end_table', 'kitchen_cabinet', 'picture', 'book', 'stool', 'coffee_table', 'bookshelf', 
              'painting', 'key_board', 'dresser', 'tv', 'whiteboard', 'cpu', 'toilet', 'file_cabinet', 'bench', 
              'ottoman', 'plant', 'monitor', 'printer', 'recycle_bin', 'door', 'fridge', 'towel', 'cup', 'mirror', 
              'laptop', 'cloth')


file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewMultiSweepImageFromFilesIndoor', sweep_num=cam_sweep_num, to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(
        type='UnifiedRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        ),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CollectUnified3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]
test_pipeline = [
    dict(type='LoadMultiViewMultiSweepImageFromFilesIndoor', sweep_num=cam_sweep_num, to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CollectUnified3D', keys=['img'])
]



data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,  #######5
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file = data_root + 'sunrgbd_infos_train_pls_ens_10c36c.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            filter_empty_gt=True,
            box_type_3d='Depth',
            file_client_args=file_client_args)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_withimg.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_withimg.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args))

evaluation = dict(pipeline=test_pipeline, interval=5)


# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
optimizer = dict(
    type='AdamW', 
    lr=1.75e-4,
    # lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))


lr_config = dict(policy='step', warmup=None, step=[32, 38])

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=40)

# fp16 setting
# fp16 = dict(loss_scale=32.)
load_from = 'faster_rcnn_r50_caffe_fpn_1x_coco_dcnv2_c.pth'

find_unused_parameters = True
