import importlib
from math import ceil
from copy import deepcopy
from .model import model
from .dataloader import dataloader
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.data.datasets import register_coco_instances
from fvcore.common.param_scheduler import MultiStepParamScheduler


def default_coco_scheduler_iteration_based(total_steps, decay_ratio=5/6, warmup_ratio=0):
    decay_steps = ceil(total_steps * decay_ratio)
    warmup_steps = ceil(warmup_ratio * decay_ratio)
    scheduler = L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[decay_steps, total_steps],
    )
    return L(WarmupParamScheduler)(
        scheduler=scheduler,
        warmup_length=warmup_ratio,
        warmup_method="linear",
        warmup_factor=0.001,
    )

NUM_ITER = 8310 # 60 epochs with batch size 8

# register custom dataset
register_coco_instances(

    "orion_train",
    {},
    "/workspace/ai_backend/dataset/dt_test/train333_utf8.json",
    "/workspace/ai_backend/dataset/dt_test/image"
    #  "/workspace/backend/dataset/base/images/"
#    /data01/backend/dataset/dt_test/val333_utf8.json
)
register_coco_instances(
    "orion_val",
    {},
    "/workspace/ai_backend/dataset/dt_test/val333_utf8.json",
    "/workspace/ai_backend/dataset/dt_test/image"
)

# import from base
optimizer = deepcopy(importlib.import_module("projects.dino.configs.dino-swin.dino_swin_base_384_4scale_12ep").optimizer)
train = deepcopy(importlib.import_module("projects.dino.configs.dino-swin.dino_swin_base_384_4scale_12ep").train)

# # modify training config
# train.init_checkpoint="/workspace/output/model/dino_swin_base_384_4scale_12ep.pth"
# train.output_dir = "/workspace/output/test_model/"

# max training iterations
train.max_iter = NUM_ITER
train.eval_period = NUM_ITER
train.log_period = 20
train.checkpointer.period = 1000000 # only save when finish training

# modify lr multiplier
lr_multiplier = default_coco_scheduler_iteration_based(NUM_ITER)



# /data01/backend/output/model/dino_swin_base_384_4scale_12ep.pth

# python detrex/tools/train_net.py --config-file /orion/config/train.py --nun-gpus 1 train.amp.enabled=False train.init_check=point=/output/model/dino_swin_basedino_swin_base_384_4scale_12ep.pth train.output_dir=output/test_model/