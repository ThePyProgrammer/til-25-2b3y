# The new config inherits a base config to highlight the necessary modification
_base_ = "../resnet_strikes_back/faster-rcnn_r50-rsb-pre_fpn_1x_coco.py"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(roi_head=dict(bbox_head=dict(num_classes=18)))

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
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train_annotations.json",
        data_prefix=dict(img="images/"),
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file="val_annotations.json",
        data_prefix=dict(img="images/"),
    )
)
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + "val_annotations.json")
test_evaluator = val_evaluator

# Set max epochs
max_epochs = 12
train_cfg = dict(max_epochs=12)
