import os
import json
import torch
import torchvision.transforms as T
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from timm.data import ImageDataset
from timm.models import create_model as timm_create_model

import torch
import torch.nn as nn
from timm.models import create_model
from projects.dn_detr.modeling import DNDetrTransformerEncoder
from detectron2.config import LazyConfig, LazyCall as L, instantiate
from detrex.layers import PositionEmbeddingSine
from detectron2.data.datasets import register_coco_instances


# register custom dataset
register_coco_instances(
    "orion_train",
    {},
    "/workspace/orion_data/annotations/annotation_train.json",
    "/workspace/orion_data/images/"
)
register_coco_instances(
    "orion_val",
    {},
    "/workspace/orion_data/annotations/annotation_valid.json",
    "/workspace/orion_data/images/"
)


cfg_dataloader = LazyConfig.load("./orion/config/dataloader.py")


# DNDetrTransformerEncoder 
# 이미지의 특성 + 위치정보결함 -> 클래스 예측
# backbone network : ResNet50(backbone)이미지특성 -> neck layer + PositionEmbeddingSine: 위치정보 -> DNDetrTransfoermerEncoder
class OrionClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.clf = timm_create_model(
            'resnet50',
            num_classes=252,
            checkpoint_path="./output/classifier/stage1/model_best.pth.tar"
        )
        self.clf.fc = torch.nn.Identity()
        self.neck = nn.Linear(2048, 256)
        self.position_embedding = PositionEmbeddingSine(
            num_pos_feats=128,
            temperature=20,
            normalize=True
        )
        self.transformer = DNDetrTransformerEncoder(
            embed_dim=256,
            num_heads=8,
            attn_dropout=0.0,
            feedforward_dim=2048,
            ffn_dropout=0.0,
            activation=nn.PReLU(),
            num_layers=3,
            post_norm=False,
        )
        self.class_embed = nn.Linear(256, 252)

    def forward(self, images, xs, ys):
        
        if len(images.size()) == 5:
            assert images.size(0) == xs.size(0) == ys.size(0) == 1
            images = images.squeeze(0)
            xs = xs.squeeze(0)
            ys = ys.squeeze(0)
        
        # Suppose image size is torch.Tensor(100, 3, 224, 224)

        # forward pass backbone and projection
        features = self.clf(images) # torch.Size([100, 2048])
        features = self.neck(features) # torch.Size([100, 256])
        
        # create mask
        bs = 1
        n, d = features.shape
        img_masks = features.new_zeros((bs, n), dtype=torch.bool) # torch.Size([1, 100])

        # create position embedding
        pe = self.position_embedding(torch.zeros(1, 128, 128, dtype=torch.bool, device=img_masks.device)) # torch.Size([1, 256, 128, 128])
        xs = (xs*128).long()
        ys = (ys*128).long()
        pos_embed = pe[0, :, ys, xs].T# torch.Size([100, 256]) model.position_embedding(img_masks) # torch.Size([3, 256, 33, 28])
        # pos_embed = torch.zeros_like(pos_embed)

        # reshape and permute for transformer
        features = features.view(bs, n, d).permute(1, 0, 2) # torch.Size([100, 1, 256])
        pos_embed = pos_embed.view(bs, n, d).permute(1, 0, 2) # torch.Size([100, 1, 256])
        # img_masks = img_masks.view(bs, -1) # torch.Size([3, 924])

        out = self.transformer(
            query=features, key=None, value=None,
            query_pos=pos_embed, query_key_padding_mask=img_masks
        ) # torch.Size([100, 1, 256])

        logits = self.class_embed(out) # torch.Size([100, 1, 252])
        
        return logits.view(n, -1) # torch.Size([100, 252])


    
def create_model(args):
    return OrionClassifier()


class OrionDataset(torch.utils.data.Dataset):
    
    def __init__(self, ds, transform=None, ignore_class=None):
        self.ds = ds
        if ignore_class is not None:
            for img in self.ds:
                img['annotations'] = [ann for ann in img['annotations'] if ann['category_id'] != ignore_class]
        self.transform = transform
        
    def __len__(self):
        return len(self.ds)
        
    def __getitem__(self, idx):
        img_info = self.ds[idx]
        img = Image.open(img_info['file_name'])
        H, W = img_info['height'], img_info['width']
        pils = []
        targets = []
        xs = []
        ys = []
        for bbox in img_info['annotations']:
            x, y, w, h = bbox['bbox']
            cropped = img.crop((x, y, x+w, y+h))
            if self.transform:
                cropped = self.transform(cropped)
            pils.append(cropped)
            targets.append(bbox['category_id'])
            xs.append((x+w/2) / W)
            ys.append((y+h/2) / H)
        return torch.stack(pils), torch.FloatTensor(xs), torch.FloatTensor(ys), torch.LongTensor(targets)      

    
def create_dataset(args, is_training=False):
    
    if is_training:
        ds = instantiate(cfg_dataloader.dataloader.train.dataset)
        return OrionDataset(ds) 
    else:
        ds = instantiate(cfg_dataloader.dataloader.test.dataset)
        return OrionDataset(ds, ignore_class=0) 


def create_transform(is_train=False):
    
    if is_train:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]),
            T.RandomErasing(p=0.4, scale=(0.33, 0.9), ratio=(0.33, 3)),
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        ])