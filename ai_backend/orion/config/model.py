import importlib
from copy import deepcopy

          
model = deepcopy(importlib.import_module("projects.dino.configs.dino-swin.dino_swin_base_384_4scale_12ep").model)
model.num_classes = 1
