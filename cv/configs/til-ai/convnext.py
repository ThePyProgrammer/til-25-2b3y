# The new config inherits a base config to highlight the necessary modification
_base_ = "../convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py"

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
