# original
import cv2
import json
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as TV
from PIL import Image
from pathlib import Path
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import instantiate
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from timm.models import create_model as timm_create_model
from projects.dn_detr.modeling import DNDetrTransformerEncoder
from detrex.layers import PositionEmbeddingSine
from . import config as cfg
import pandas as pd


#객체 탐지를 위한 모델 로드 
#입력된 이미지에 대해 객체 탐지 수행, 탐지된 객체의 위치(bbox),score,class ID 반환

class OneStageOrionDetector:
    
    def __init__(self, checkpoint):
        with (Path(__file__).parent / "class_info.json").open('r') as f:
            class_info = json.load(f) #클래스 정보를 json파일에서 읽어옴. 
            self.class_info = {int(k): class_info[k] for k in class_info}
            # 클래스정보를 딕셔너리형태로 저장. 
        self.augmentation = instantiate(cfg.dataloader.test.mapper.augmentation)   
        self.model = instantiate(cfg.model).to('cuda')
        self.model.eval()
        DetectionCheckpointer(self.model).load(checkpoint)
        #모델 체크포인트 로드. 
        for _ in range(3):
            self.warmup()
# 모델 출력에서 임계 값 이상의 객체만 추출함. 
    def __call__(self, image, threshold=0.3):

        dataset_dict = {}
        utils.check_image_size(dataset_dict, image)
        image, _ = T.apply_transform_gens(self.augmentation, image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        with torch.no_grad():
            output = self.model([dataset_dict])[0]['instances']

        confidence = output.get('scores')
        output = output[confidence >= threshold]
        output = output.to('cpu')

        boxes = output.pred_boxes.tensor.tolist()
        scores = output.scores.tolist()
        preds = output.pred_classes.tolist()

        return boxes, scores, preds

    def warmup(self):
        img = cv2.imread("example.jpg")[...,::-1]
        self(img)

# detection에서 탐지된 객체들에 대하여 클래스 예측 
class OrionClassifier(nn.Module):
    # resnet50 기반의 분류기 정의, 이미지 특성과 위치 정보를 기반으로 클래스 예측. 
    def __init__(self, class_num):
        super().__init__()
        self.clf = timm_create_model(
            'resnet50',
            num_classes=class_num
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
        self.class_embed = nn.Linear(256, class_num)
# 이미지의 특성 추출, 위치정보 
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

# 탐지 및 분류 모델을 결합. 
class DetwithClfContext:
    
    """Detrex Stye Input Ouput 2Stage Detector"""
    
    def __init__(self, det, clf):
        self.det = det
        self.clf = clf
        self.transform = TV.Compose([
            TV.Resize((224, 224)),
            TV.ToTensor(),
            TV.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        ])
# det_thr 객체의 신뢰도 점수 필터링, det_thr이상의 신뢰도 점수를 가진 객체만 후속처리(분류)를 위해 선택
# clf_thr 이상의 확률을 가진 클래스 만이 최종 분류결과로 선택 됨.
    def __call__(self, input, det_thr=0.7, clf_thr=0.1):
        
        assert len(input) == 1 # only batch size 1 is allowed
        
        img_info = input[0]
        if 'pil_image' in img_info:
            img = img_info['pil_image']
        else:
            img = Image.open(img_info['file_name'])
        H, W = img_info['height'], img_info['width']
        
        clf_inputs = []
        xs = []
        ys = []

        output = self.det(input)[0]['instances']
        output = output[output.get('scores') >= det_thr]
        
        for x1, y1, x2, y2 in output.get('pred_boxes').tensor.cpu().numpy():
            cropped = img.crop((x1, y1, x2, y2))
            if self.transform:
                cropped = self.transform(cropped)
            clf_inputs.append(cropped)
            xs.append((x1 + x2) / (2*W))
            ys.append((y1 + y2) / (2*H))
        
        if not clf_inputs: # empty list
            return [{'instances': output}]

        clf_inputs = torch.stack(clf_inputs).cuda()
        xs = torch.FloatTensor(xs).cuda()
        ys = torch.FloatTensor(ys).cuda()
        clf_outputs = self.clf(clf_inputs, xs, ys)
        clf_outputs = torch.softmax(clf_outputs, dim=-1)
        conf, preds = clf_outputs.max(dim=-1)
        preds[conf < clf_thr] = 0
 
        output.set('pred_classes', preds)
        output.set('scores', conf)
        
        return [{'instances': output}]


class TwoStageOrionDetector:

    def __init__(self, det_checkpoint, clf_checkpoint, class_info_path ,class_num):
        self.class_info = pd.read_excel(class_info_path,index_col=0).fillna('nan').T.to_dict()
        self.augmentation = instantiate(cfg.dataloader.test.mapper.augmentation)
        
        # load detector checkpoint
        det = instantiate(cfg.model).to('cuda')
        det.eval()
        DetectionCheckpointer(det).load(det_checkpoint)
        
        # load classifier checkpoint
        clf = OrionClassifier(class_num)
        clf.load_state_dict(torch.load(clf_checkpoint)['state_dict'])
        clf.eval()
        clf.cuda()
        
        self.model = DetwithClfContext(det, clf)

        # for _ in range(3):
        #     self.warmup()

    def __call__(self, image, threshold=0.3, clf_threshold=0.7):
        #logger.info("TwoStageOrionDetector __call__ method invoked")
        np_image = np.asarray(image)
        dataset_dict = {}
        utils.check_image_size(dataset_dict, np_image)
        np_image, _ = T.apply_transform_gens(self.augmentation, np_image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(np_image.transpose(2, 0, 1)))
        dataset_dict["pil_image"] = image

        with torch.no_grad():
            output = self.model([dataset_dict], threshold, clf_threshold)[0]['instances']

        output = output.to('cpu')

        boxes = output.pred_boxes.tensor.tolist()
        scores = output.scores.tolist()
        preds = output.pred_classes.tolist()

        return boxes, scores, preds
    
'''
    def warmup(self):
        img = Image.open("example.jpg")
        self(img)
''' 
