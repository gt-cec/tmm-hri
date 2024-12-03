# _base_ = ['../../../_base_/default_runtime.py']

import os
SEQ_LEN = 1 # Default to 27 if not set
print("Sequence length choice", SEQ_LEN)

REFORMULATION_ADDITIONAL_INPUTS = ''  # Default value if not set
BATCH_SIZE = 128

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Pose3dLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# runtime
train_cfg = dict(max_epochs=160, val_interval=10)

# optimizer
optim_wrapper = dict(optimizer=dict(type='Adam', lr=1e-3))

# learning policy
param_scheduler = [
    dict(type='ExponentialLR', gamma=0.975, end=80, by_epoch=True)
]

auto_scale_lr = dict(base_batch_size=1024)

# hooks
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='MPJPE',
        rule='less',
        max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=20),
)

# codec settings
codec = dict(
    type='VideoPoseLifting',
    num_keypoints=17,
    zero_center=True,
    root_index=0,
    remove_root=False)

configs = {
    'BL_OD_RT': 5 * 17,  # Bone Lengths, Ordinal Depth, Rotation/Translation
    'BL_OD': 4 * 17,     # Bone Lengths and Ordinal Depth
    'BL': 3 * 17,        # Bone Lengths only
    'OD': 3 * 17,        # Ordinal Depth only
    'RT': 3 * 17,        # Rotation/Translation only
    '': 2 * 17           # Empty case
}

NUM_CHANNELS = configs.get(REFORMULATION_ADDITIONAL_INPUTS, 2 * 17)  # Default to 2 * 17 if not found
print(f"Total number of channels needed for '{REFORMULATION_ADDITIONAL_INPUTS}': {NUM_CHANNELS}")


# Select the model configuration based on SEQ_LEN
if SEQ_LEN == 1:
    model = dict(
        type='PoseLifter',
        backbone=dict(
            type='TCN',
            in_channels=NUM_CHANNELS,
            stem_channels=1024,
            num_blocks=4,
            kernel_sizes=(1, 1, 1, 1, 1),
            dropout=0.25,
            use_stride_conv=True,
        ),
        head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss=dict(type='MPJPELoss'),
            decoder=codec,
        ),
    )
elif SEQ_LEN == 3:
    model = dict(
        type='PoseLifter',
        backbone=dict(
            type='TCN',
            in_channels=NUM_CHANNELS,
            stem_channels=1024,
            num_blocks=2,
            kernel_sizes=(3, 1, 1),
            dropout=0.25,
            use_stride_conv=True,
        ),
        head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss=dict(type='MPJPELoss'),
            decoder=codec,
        )
    )
elif SEQ_LEN == 9:
    model = dict(
        type='PoseLifter',
        backbone=dict(
            type='TCN',
            in_channels=NUM_CHANNELS,
            stem_channels=1024,
            num_blocks=2,
            kernel_sizes=(3, 3, 1),
            dropout=0.25,
            use_stride_conv=True,
        ),
        head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss=dict(type='MPJPELoss'),
            decoder=codec,
        )
    )
elif SEQ_LEN == 27:
    model = dict(
        type='PoseLifter',
        backbone=dict(
            type='TCN',
            in_channels=NUM_CHANNELS,
            stem_channels=1024,
            num_blocks=2,
            kernel_sizes=(3, 3, 3),
            dropout=0.25,
            use_stride_conv=True,
        ),
        head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss=dict(type='MPJPELoss'),
            decoder=codec,
        )
    )
elif SEQ_LEN == 81:
    model = dict(
        type='PoseLifter',
        backbone=dict(
            type='TCN',
            in_channels=NUM_CHANNELS,
            stem_channels=1024,
            num_blocks=3,
            kernel_sizes=(3, 3, 3, 3),
            dropout=0.25,
            use_stride_conv=True,
        ),
        head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss=dict(type='MPJPELoss'),
            decoder=codec,
        ),
    )
elif SEQ_LEN == 243:
    model = dict(
        type='PoseLifter',
        backbone=dict(
            type='TCN',
            in_channels=NUM_CHANNELS,
            stem_channels=1024,
            num_blocks=4,
            kernel_sizes=(3, 3, 3, 3, 3),
            dropout=0.25,
            use_stride_conv=True,
        ),
        head=dict(
            type='TemporalRegressionHead',
            in_channels=1024,
            num_joints=17,
            loss=dict(type='MPJPELoss'),
            decoder=codec,
        ),
    )
else:
    raise ValueError(f"Unsupported SEQ_LEN: {SEQ_LEN}")



# base dataset settings
dataset_type = 'Fit3D_Dataset'
data_root = 'data/h36m/'

# pipelines
train_pipeline = [
    dict(
        type='RandomFlipAroundRoot',
        keypoints_flip_cfg=dict(),
        target_flip_cfg=dict(),
    ),
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root'))
]
val_pipeline = [
    dict(type='GenerateTarget', encoder=codec),
    dict(
        type='PackPoseInputs',
        meta_keys=('id', 'category_id', 'target_img_path', 'flip_indices',
                   'target_root'))
]

# data loaders
train_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file='annotation_body3d/fps50/h36m_train.npz',
        seq_len=SEQ_LEN,
        causal=False,
        pad_video_seq=True,
        camera_param_file='annotation_body3d/cameras.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=BATCH_SIZE,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=True, round_up=False),
    dataset=dict(
        type=dataset_type,
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_validation_.npz',
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_validation__25hz.npz', # use this one for paper evals
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_validation__25hz_10_23_novel_exc_analysis.npz', # this is the diff split based on top 7 worst actions
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_validation__dbg.npz', #subsampled for debugging
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_validation__dbg_9_13_random_transforms_discrete.npz',
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_validation__25hz_reversed.npz',
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_validation__50hz_novel_exc_analysis.npz',
        # ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_all__50hz_combine_train_al.npz',
        ann_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/fit3d_annotations_all__25hz_combine_train_all.npz',
        seq_len=SEQ_LEN,
        causal=False,
        pad_video_seq=True,
        # camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_25hzcameras.pkl',
        # camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_25hz_10_23_novel_exc_analysiscameras.pkl',
        # camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_25hz_reversedcameras.pkl',
        # camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_50hzcameras.pkl',
        # camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_50hz_novel_exc_analysiscameras.pkl',
        # camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_50hz_combine_train_alcameras.pkl',
        camera_param_file='/srv/essa-lab/flash3/nwarner30/pose_estimation/fit3d_data/processed/_25hz_combine_train_allcameras.pkl',
        data_root=data_root,
        data_prefix=dict(img='images/'),
        pipeline=val_pipeline,
        test_mode=True,
    ))
test_dataloader = val_dataloader

# evaluators
val_evaluator = [
    dict(type='MPJPE', mode='mpjpe', dataset_type='fit3d'),
    dict(type='MPJPE', mode='n-mpjpe', dataset_type='fit3d'),
    dict(type='MPJPE', mode='p-mpjpe', dataset_type='fit3d'),
]
test_evaluator = val_evaluator
