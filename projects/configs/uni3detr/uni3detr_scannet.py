_base_ = [
    '../../../configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'


voxel_size = [0.02, 0.02, 0.02]
grid_size = [128, 640, 640]

point_cloud_range = [-6.4, -6.4, -0.1, 6.4, 6.4, 2.46]


fp16_enabled = True
bev_stride = 4
sample_num = 5

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='Uni3DETR',
    pts_voxel_layer=dict(
        max_num_points=5, voxel_size=voxel_size, max_voxels=(16000, 40000),  ###16000
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoderHD',
        in_channels=4,
        sparse_shape=grid_size,
        output_channels=256,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock',
        fp16_enabled=False), # not enable FP16 here
    pts_backbone=dict(
        type='SECOND3D',
        in_channels=[256, 256, 256],
        out_channels=[128, 256, 512],
        layer_nums=[5, 5, 5],
        layer_strides=[1, 2, 4],
        is_cascade=False,
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv3d', kernel=(1,3,3), bias=False)),
    pts_neck=dict(
        type='SECOND3DFPN',
        in_channels=[128, 256, 512],
        out_channels=[256, 256, 256],
        upsample_strides=[1, 2, 4],
        norm_cfg=dict(type='BN3d', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv3d', bias=False),
        extra_conv=dict(type='Conv3d', num_conv=3, bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='Uni3DETRHead',
        # transformer_cfg
        num_query=300,
        num_classes=18,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=8,
        with_nms=True,
        transformer=dict(
            type='Uni3DETRTransformer',
            fp16_enabled=fp16_enabled,
            decoder=dict(
                type='Uni3DETRTransformerDecoder',
                num_layers=3,
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
                            num_sweeps=1,
                            fp16_enabled=fp16_enabled),
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=512,
                        num_fcs=2,
                        ffn_drop=0.1, ##0.1
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    norm_cfg=dict(type='LN'),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))
                    # operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm'))
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            # max_num=1000,
            max_num=5000,
            voxel_size=voxel_size,
            num_classes=18), 
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(type='SoftFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.5),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='IoU3DLoss', loss_weight=1.2), 
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ),
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
            iou_cost=dict(type='IoU3DCost', weight=1.2),
            pc_range=point_cloud_range))))


dataset_type = 'ScanNetDataset'
data_root = './data/scannet/'
class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(type='LoadAnnotations3D'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='PointSample', num_points=200000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        load_dim=3,
        use_dim=[0, 1, 2]),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=3, ##### 3
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=6,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')
        ),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

evaluation = dict(pipeline=test_pipeline, interval=5)


# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 2e-5 *2/8 * 20 * 4/6  *6/8  *1.5 *8/6###########40 # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))


lr_config = dict(policy='step', warmup=None, step=[32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)  ###40

# fp16 setting
fp16 = dict(loss_scale=32.)
find_unused_parameters = True
