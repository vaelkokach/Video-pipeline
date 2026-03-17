# TSM-MobileNetV2: lightweight action recognition (~2.7M params, 3.3G FLOPs vs TSN-R50 24M/33G)
_base_ = [
    '../../_base_/models/tsm_mobilenet_v2.py',
    '../../_base_/default_runtime.py',
]

file_client_args = dict(io_backend='disk')

# Required by MMAction2 inference_recognizer
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True,
    ),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs'),
]

load_from = None
resume = False
