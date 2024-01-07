_base_ = [
    './_base_/default_runtime.py'
]

# model type
type = 'Mapper'
plugin = True

# plugin code dir
plugin_dir = 'plugin/'

# img configs
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

img_h = 480
img_w = 800
img_size = (img_h, img_w)

num_gpus = 8
batch_size = 4
num_iters_per_epoch = 27846 // (num_gpus * batch_size)
num_epochs = 30
total_iters = num_iters_per_epoch * num_epochs

num_queries = 100

# category configs
cat2id = {
    'ped_crossing': 0,
    'divider': 1,
    'boundary': 2,
}
num_class = max(list(cat2id.values())) + 1

# bev configs
roi_size = (60, 30)
bev_h = 50
bev_w = 100
pc_range = [-roi_size[0]/2, -roi_size[1]/2, -3, roi_size[0]/2, roi_size[1]/2, 5]

# vectorize params
coords_dim = 2
sample_dist = -1
sample_num = -1
simplify = True

# meta info for submission pkl
meta = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False,
    output_format='vector')

# model configs
bev_embed_dims = 256
embed_dims = 512
num_feat_levels = 3
norm_cfg = dict(type='BN2d')
num_class = max(list(cat2id.values()))+1
num_points = 20
permute = True


## sparse4d model
task_config = dict(
    with_det = False,
    with_map = True,
    with_motion = False,
)
num_vector = 100
num_sample = 20
num_single_frame_decoder = 0
num_decoder = 6
drop_out = 0.1
embed_dims = 256
num_groups = 8
num_levels = 3
model = dict(
    type="Sparse4D",
    use_grid_mask=True,
    use_deformable_func=True,
        img_backbone=dict(
            type='ResNet',
            with_cp=False,
            pretrained='open-mmlab://detectron2/resnet50_caffe',
            depth=50,
            num_stages=4,
            out_indices=(1, 2, 3),
            frozen_stages=-1,
            norm_cfg=norm_cfg,
            norm_eval=True,
            style='caffe',
            dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True)
        ),
        img_neck=dict(
            type='FPN',
            in_channels=[512, 1024, 2048],
            out_channels=256,
            start_level=0,
            add_extra_convs=True,
            num_outs=3,
            norm_cfg=norm_cfg,
            relu_before_extra_convs=True
        ),
    head=dict(
        type="Sparse4DHead",
        task_config=task_config,
        map_head=dict(
            type="Sparse4DMapHead",
            cls_threshold_to_reg=0.05,
            instance_bank=dict(
                type="InstanceBank",
                num_anchor=100,
                embed_dims=embed_dims,
                anchor="kmeans_map_100.npy",
                anchor_handler=dict(type="SparsePoint3DKeyPointsGenerator"),
                num_temp_instances=0,
                confidence_decay=0.6,
                feat_grad=True,
            ),
            anchor_encoder=dict(
                type="SparsePoint3DEncoder",
                embed_dims=embed_dims,
                num_sample=num_sample,
            ),
            num_single_frame_decoder=num_single_frame_decoder,
            operation_order=[
                "deformable",
                "ffn",
                "norm",
                "refine",
            ] * num_single_frame_decoder + [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "ffn",
                "norm",
                "refine",
            ] * (num_decoder - num_single_frame_decoder),
            temp_graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            graph_model=dict(
                type="MultiheadFlashAttention",
                embed_dims=embed_dims,
                num_heads=num_groups,
                batch_first=True,
                dropout=drop_out,
            ),
            norm_layer=dict(type="LN", normalized_shape=embed_dims),
            ffn=dict(
                type="AsymmetricFFN",
                in_channels=embed_dims * 2,
                pre_norm=dict(type="LN"),
                embed_dims=embed_dims,
                feedforward_channels=embed_dims * 4,
                num_fcs=2,
                ffn_drop=drop_out,
                act_cfg=dict(type="ReLU", inplace=True),
            ),
            deformable_model=dict(
                type="DeformableFeatureAggregation",
                embed_dims=embed_dims,
                num_groups=num_groups,
                num_levels=num_levels,
                num_cams=6,
                attn_drop=0.15,
                use_deformable_func=True,
                use_camera_embed=True,
                residual_mode="cat",
                kps_generator=dict(
                    type="SparsePoint3DKeyPointsGenerator",
                    embed_dims=embed_dims,
                    num_sample=num_sample,
                    num_learnable_pts=3,
                    fix_height=(0, 0.5, -0.5, 1, -1),
                    ground_height=0, # ground height in lidar frame
                ),
            ),
            refine_layer=dict(
                type="SparsePoint3DRefinementModule",
                embed_dims=embed_dims,
                num_sample=num_sample,
                num_cls=3,
            ),
            sampler=dict(
                type="SparsePoint3DTarget",
                assigner=dict(
                    type='HungarianLinesAssigner_v2',
                    cost=dict(
                        type='MapQueriesCost',
                        cls_cost=dict(type='FocalLossCost', weight=4.0),
                        reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=True),
                    ),
                ),
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=4.0,
            ),
            loss_reg=dict(
                type='LinesL1Loss',
                loss_weight=50.0,
                beta=0.01,
            ),
            gt_cls_key="gt_map_labels",
            gt_reg_key="gt_map_pts",
            decoder=dict(type="SparsePoint3DDecoder"),
            reg_weights=[1.0] * 40,
            roi_size=roi_size,
            num_sample=num_sample,
        ),
    ),
)

# data processing pipelines
train_pipeline = [
    dict(
        type='VectorizeMap_v2',
        coords_dim=coords_dim,
        roi_size=roi_size,
        sample_num=num_points,
        normalize=True,
        permute=permute,
        
    ),
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='CustomFormatBundle3D'),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(
        type='Collect3D', 
        keys=[            
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
            'gt_map_labels', 
            'gt_map_pts',
        ], 
        meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name', "T_global", "T_global_inv", "timestamp"
        ),
    ),
]

# data processing pipelines
test_pipeline = [
    dict(type='LoadMultiViewImagesFromFiles', to_float32=True),
    dict(type='ResizeMultiViewImages',
         size=img_size, # H, W
         change_intrinsics=True,
         ),
    dict(type='Normalize3D', **img_norm_cfg),
    dict(type='PadMultiViewImages', size_divisor=32),
    dict(type='CustomFormatBundle3D'),
    dict(type="NuScenesSparse4DAdaptor"),
    dict(type='Collect3D', 
        keys=[
            "img",
            "timestamp",
            "projection_mat",
            "image_wh",
        ],
        meta_keys=(
        'token', 'ego2img', 'sample_idx', 'ego2global_translation',
        'ego2global_rotation', 'img_shape', 'scene_name', "T_global", "T_global_inv", "timestamp"))
]

# configs for evaluation code
# DO NOT CHANGE
data_root = './data/nuscenes'
ann_root = './data/nuscenes_cam/'
eval_config = dict(
    type='NuscDataset',
    data_root=data_root,
    ann_file=ann_root+'nuscenes_map_infos_val.pkl',
    meta=meta,
    roi_size=roi_size,
    cat2id=cat2id,
    pipeline=[
        dict(
            type='VectorizeMap',
            coords_dim=coords_dim,
            simplify=True,
            normalize=False,
            roi_size=roi_size
        ),
        dict(type='CustomFormatBundle3D'),
        dict(type='Collect3D', keys=['vectors'], meta_keys=['token', 'timestamp'])
    ],
    interval=1,
)

# dataset configs
data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=4,
    train=dict(
        type='NuscDataset',
        data_root=data_root,
        ann_file=ann_root+'nuscenes_map_infos_train.pkl',
        map_ann_file='tmp_gts_nusc_60x30_train.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=train_pipeline,
        seq_split_num=-1,
    ),
    val=dict(
        type='NuscDataset',
        data_root=data_root,
        ann_file=ann_root+'nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=-1,
    ),
    test=dict(
        type='NuscDataset',
        data_root=data_root,
        ann_file=ann_root+'nuscenes_map_infos_val.pkl',
        meta=meta,
        roi_size=roi_size,
        cat2id=cat2id,
        pipeline=test_pipeline,
        eval_config=eval_config,
        test_mode=True,
        seq_split_num=-1,
    ),
    shuffler_sampler=dict(
        type='InfiniteGroupEachSampleInBatchSampler',
    ),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

# optimizer
optimizer = dict(
    type='AdamW',
    lr=5e-4 * (batch_size / 4),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

# learning policy & schedule
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=3e-3)

evaluation = dict(interval=num_epochs//6*num_iters_per_epoch)
find_unused_parameters = True #### when use checkpoint, find_unused_parameters must be False
checkpoint_config = dict(interval=num_epochs//6*num_iters_per_epoch)

runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

SyncBN = True