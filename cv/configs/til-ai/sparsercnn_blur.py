# The new config inherits a base config to highlight the necessary modification
_base_ = "../sparse_rcnn/sparse-rcnn_r50_fpn_300-proposals_crop-ms-480-800-3x_coco.py"

# add mmpretrain transforms
custom_imports = dict(
    imports=["mmpretrain.datasets.transforms"], allow_failed_imports=False
)

# We also need to change the num_classes in head to match the dataset's annotation
num_classes = 18
bbox_head = _base_.model.roi_head.bbox_head

for head in bbox_head:
    head.num_classes = num_classes

model = dict(roi_head=dict(bbox_head=bbox_head))

# Modify dataset related settings
data_root = "../cv/"
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

train_pipeline = _base_.train_pipeline
train_pipeline[1].with_mask = False
train_pipeline.insert(
    4,
    dict(
        type="mmpretrain.Albumentations",
        transforms=[
            dict(
                type="OneOf",
                transforms=[
                    dict(type="Blur", blur_limit=(3, 7)),
                    dict(type="MedianBlur", blur_limit=(3, 7)),
                    dict(type="Defocus", radius=(3, 10), alias_blur=(0.1, 0.5)),
                ],
                p=1,
            ),
        ],
    ),
)

val_pipeline = _base_.test_pipeline.copy()
val_pipeline[2].with_mask = False
val_pipeline.insert(
    3,
    dict(
        type="mmpretrain.Albumentations",
        transforms=[
            dict(
                type="OneOf",
                transforms=[
                    dict(type="Blur", blur_limit=(3, 7)),
                    dict(type="MedianBlur", blur_limit=(3, 7)),
                    dict(type="Defocus", radius=(3, 10), alias_blur=(0.1, 0.5)),
                ],
                p=1,
            ),
        ],
    ),
)

test_pipeline = _base_.test_pipeline.copy()
test_pipeline[2].with_mask = False

train_dataloader = dict(
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train_annotations.json",
        data_prefix=dict(img="images/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val_annotations.json",
        data_prefix=dict(img="images/"),
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=3,
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
max_epochs = 10
train_cfg = dict(max_epochs=10)
