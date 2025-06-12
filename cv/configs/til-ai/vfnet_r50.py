# The new config inherits a base config to highlight the necessary modification
_base_ = "../vfnet/vfnet_r50-mdconv-c3-c5_fpn_ms-2x_coco.py"

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(bbox_head=dict(num_classes=18))

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
max_epochs = 3
train_cfg = dict(max_epochs=3)

# load pretrained model
load_from = "https://download.openmmlab.com/mmdetection/v2.0/vfnet/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco/vfnet_r50_fpn_mdconv_c3-c5_mstrain_2x_coco_20201027pth-6879c318.pth"  # noqa
