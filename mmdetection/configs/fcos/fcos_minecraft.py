"""FCOS fine-tuning configuration for the Minecraft mobs dataset."""

_base_ = [
    "../_base_/models/fcos_r50_caffe_fpn_gn-head.py",
    "../_base_/datasets/coco_detection.py",
    "../_base_/default_runtime.py",
    "../_base_/schedules/schedule_1x.py",
]

metainfo = dict(
    classes=(
        "bee",
        "chicken",
        "cow",
        "creeper",
        "enderman",
        "fox",
        "frog",
        "ghast",
        "goat",
        "llama",
        "pig",
        "sheep",
        "skeleton",
        "spider",
        "turtle",
        "wolf",
        "zombie",
    )
)

img_scale = (512, 512)
data_root = "datasets/minecraft/"

model = dict(bbox_head=dict(num_classes=len(metainfo["classes"])))

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    pin_memory=False,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/train_annotations.coco.json",
        data_prefix=dict(img="train/images/"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/val_annotations.coco.json",
        data_prefix=dict(img="val/images/"),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="annotations/test_annotations.coco.json",
        data_prefix=dict(img="test/images/"),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    ann_file=data_root + "annotations/val_annotations.coco.json",
    metric="bbox",
    format_only=False,
)

test_evaluator = dict(
    ann_file=data_root + "annotations/test_annotations.coco.json",
    metric="bbox",
)

optim_wrapper = dict(
    optimizer=dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001),
    clip_grad=None,
)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=12, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=3, save_best="coco/bbox_mAP"),
    logger=dict(type="LoggerHook", interval=50),
)

fp16 = dict(loss_scale="dynamic")

work_dir = "artifacts/fcos"

auto_scale_lr = dict(enable=False)
