# The new config inherits a base config to highlight the necessary modification
_base_ = "./convnext2_tiny.py"

# add albumentations with bbox support
custom_imports = dict(
    imports=["mmdetection.configs.til-ai.albumentations_fix"],
    allow_failed_imports=False,
)

# We also need to change the num_classes in head to match the dataset's annotation
num_classes = 18
bbox_head = _base_.model.roi_head.bbox_head

for head in bbox_head:
    head.num_classes = num_classes

model = dict(
    roi_head=dict(bbox_head=bbox_head)
)

# Modify dataset related settings
data_root = "/cv/"
metainfo = {
    "classes": (
        "cargo aircraft",
        "commercial aircraft",
        "drone",
        "fighter jet",
        "fighter plane",
        "helicopter",
        "light aircraft",
        "missile",
        "truck",
        "car",
        "tank",
        "bus",
        "van",
        "cargo ship",
        "yacht",
        "cruise ship",
        "warship",
        "sailboat",
    )
}

image_size = (1024, 1024)

train_pipeline = [
    dict(type="MixUp", img_scale=(1920, 1080)),
    dict(
        transforms=[
            dict(
                p=0.75,
                rotate=(
                    -30.0,
                    30.0,
                ),
                scale=(
                    0.5,
                    1.0,
                ),
                type="Affine",
            ),
            dict(
                p=0.25,
                transforms=[
                    dict(type="ToGray"),
                    dict(type="ChannelDropout"),
                ],
                type="OneOf",
            ),
            dict(
                p=0.75,
                transforms=[
                    dict(
                        type="Blur",blur_limit=(5, 10)
                    ),
                    dict(type="MedianBlur", blur_limit=(5, 10)),
                    dict(type="Defocus", radius=(5, 15), alias_blur=(0.1, 0.3)),
                ],
                type="OneOf",
            ),
        ],
        type="mmpretrain.AlbuHBBoxes",
    ),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(prob=0.5, type="RandomFlip"),
    dict(type="PackDetInputs"),
]

train_dataset = dict(
    _delete_=True,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train_annotations.json",
        data_prefix=dict(img="images/"),
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
        ],
    ),
    pipeline=train_pipeline,
    type="MultiImageMixDataset",
)

test_pipeline = _base_.test_pipeline.copy()
test_pipeline[2].with_mask = False

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=train_dataset,
)

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val_annotations.json",
        data_prefix=dict(img="images/"),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val_annotations.json",
        data_prefix=dict(img="images/"),
        pipeline=test_pipeline,
    ),
)

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + "val_annotations.json", metric=["bbox"])
test_evaluator = val_evaluator
