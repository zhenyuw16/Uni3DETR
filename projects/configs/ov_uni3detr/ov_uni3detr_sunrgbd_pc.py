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

fp16_enabled = False
bev_stride = 4
sample_num = 5


input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False)

model = dict(
    type='OV_Uni3DETR',
    pts_voxel_layer=dict(
       max_num_points=5, voxel_size=voxel_size, max_voxels=(16000, 40000),
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
                        ffn_drop=0.1,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),
                    norm_cfg=dict(type='LN'),
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm'))
            )
        ),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=point_cloud_range,
            pc_range=point_cloud_range,
            max_num=1000,
            voxel_size=voxel_size, 
            alpha=1.0,
            num_classes=46), 
        post_processing=dict(
            type='nms',
            nms_thr=0.5),
        ######## soft nms can generate a little higher result
        # post_processing=dict(
        #     type='soft_nms',
        #     gaussian_sigma=0.3, 
        #     prune_threshold=1e-2),
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


dataset_type = 'SUNRGBDDataset_OV'
data_root = 'data/sunrgbd_coda/'
class_names = ('chair', 'table', 'pillow', 'sofa_chair', 'desk', 'bed', 'sofa', 'computer', 'box', 
              'lamp', 'garbage_bin', 'cabinet', 'shelf', 'drawer', 'sink', 'night_stand', 'kitchen_counter', 
              'paper', 'end_table', 'kitchen_cabinet', 'picture', 'book', 'stool', 'coffee_table', 'bookshelf', 
              'painting', 'key_board', 'dresser', 'tv', 'whiteboard', 'cpu', 'toilet', 'file_cabinet', 'bench', 
              'ottoman', 'plant', 'monitor', 'printer', 'recycle_bin', 'door', 'fridge', 'towel', 'cup', 'mirror', 
              'laptop', 'cloth')

seen_classes = ('chair', 'table', 'pillow', 'sofa_chair', 'desk', 'bed', 'sofa', 'computer', 'lamp', 'box')

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D'),
    dict(
        type='UnifiedRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
    ),
    dict(
        type='UnifiedRotScaleTrans',
        rot_range=[-0.523599, 0.523599],
        scale_ratio_range=[0.85, 1.15],
        shift_height=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='PointSample', num_points=20000),
    dict(type='PointSample', num_points=200000),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=True,
        load_dim=6,
        use_dim=[0, 1, 2],
        file_client_args=file_client_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='PointSample', num_points=50000),
    dict(type='PointSample', num_points=200000),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,  #######5
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train_pls_ens_10c36c.pkl',
            pipeline=train_pipeline,
            classes=class_names,
            seen_classes=seen_classes,
            filter_empty_gt=True,
            box_type_3d='Depth',
            file_client_args=file_client_args)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_withimg.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        seen_classes=seen_classes,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_withimg.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        seen_classes=seen_classes,
        test_mode=True,
        box_type_3d='Depth',
        file_client_args=file_client_args))

evaluation = dict(pipeline=test_pipeline, interval=5)


# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
lr = 2e-5 *2/8 * 40 # max learning rate
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))


lr_config = dict(policy='step', warmup=None, step=[32, 38])
runner = dict(type='EpochBasedRunner', max_epochs=40)

# fp16 setting
# fp16 = dict(loss_scale=32.)
find_unused_parameters = True
