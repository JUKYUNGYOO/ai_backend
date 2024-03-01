import importlib
from copy import deepcopy
from omegaconf import ListConfig, DictConfig


def multiply_resolution_dict(cfg, n):
    
    if cfg is None:
        return 
    
    for k in cfg:
        if k in ('short_edge_length', 'max_size', 'crop_size'):
            if isinstance(cfg[k], int):
                cfg[k] *= n
            elif isinstance(cfg[k], (list, ListConfig)):
                cfg[k] = ListConfig([x*n for x in cfg[k]])
        elif isinstance(cfg[k], ListConfig):
            for v in cfg[k]:
                multiply_resolution_dict(v, n)
        elif isinstance(cfg[k], DictConfig):
            multiply_resolution_dict(cfg[k], n)

            
dataloader = deepcopy(importlib.import_module("projects.dino.configs.dino-swin.dino_swin_base_384_4scale_12ep").dataloader)

multiply_resolution_dict(dataloader, 1)
dataloader.test.dataset.names = 'orion_val'
dataloader.train.dataset.names = 'orion_train'
dataloader.evaluator.max_dets_per_image=1000
dataloader.train.total_batch_size = 1
dataloader.train.num_workers = 4