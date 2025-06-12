# The new config inherits a base config to highlight the necessary modification
_base_ = "../convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py"


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
    roi_head=dict(bbox_head=bbox_head, mask_head=None, mask_roi_extractor=None),
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

train_pipeline = [
    dict(type="MixUp", img_scale=(1920, 1080)),
    dict(prob=0.5, type="RandomFlip"),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type="RandomChoiceResize",
                ),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            1333,
                        ),
                        (
                            500,
                            1333,
                        ),
                        (
                            600,
                            1333,
                        ),
                    ],
                    type="RandomChoiceResize",
                ),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type="absolute_range",
                    type="RandomCrop",
                ),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            1333,
                        ),
                        (
                            512,
                            1333,
                        ),
                        (
                            544,
                            1333,
                        ),
                        (
                            576,
                            1333,
                        ),
                        (
                            608,
                            1333,
                        ),
                        (
                            640,
                            1333,
                        ),
                        (
                            672,
                            1333,
                        ),
                        (
                            704,
                            1333,
                        ),
                        (
                            736,
                            1333,
                        ),
                        (
                            768,
                            1333,
                        ),
                        (
                            800,
                            1333,
                        ),
                    ],
                    type="RandomChoiceResize",
                ),
            ],
        ],
        type="RandomChoice",
    ),
    dict(
        transforms=[
            dict(
                p=0.7,
                rotate=(
                    -15.0,
                    15.0,
                ),
                scale=(
                    0.8,
                    1.2,
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
                        type="Blur",
                    ),
                    dict(
                        type="MedianBlur",
                    ),
                    dict(
                        type="Defocus",
                    ),
                ],
                type="OneOf",
            ),
        ],
        type="mmpretrain.AlbuHBBoxes",
    ),
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

# Set max epochs
max_epochs = 12
train_cfg = dict(max_epochs=12)
