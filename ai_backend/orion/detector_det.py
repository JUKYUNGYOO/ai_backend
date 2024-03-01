# detection - detector
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
from . import config as cfg
import pandas as pd

class OneStageOrionDetector:
    
    def __init__(self, checkpoint):
        with (Path(__file__).parent / "class_info.json").open('r') as f:
            class_info = json.load(f)
            # 클래스 info 사용 안함. 
            # self.class_info = {int(k): v for k, v in class_info.items()}
        self.augmentation = instantiate(cfg.dataloader.test.mapper.augmentation)   
        self.model = instantiate(cfg.model).to('cuda')
        self.model.eval()
        DetectionCheckpointer(self.model).load(checkpoint)
        for _ in range(3):
            self.warmup()

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

      
        # 클래스 정보를 무시하고 bbox 정보만 생성
        boxes = output.pred_boxes.tensor.tolist()
        scores = output.scores.tolist()
        #  preds = output.pred_classes.tolist()
        preds = [0] * len(boxes)  # 모든 클래스를 bbox로 설정

        return boxes, scores, preds

    def warmup(self):
        img = cv2.imread("example.jpg")[...,::-1]
        self(img)

class TwoStageOrionDetector:

    def __init__(self, det_checkpoint):
        with (Path(__file__).parent / "class_info.json").open('r') as f:
            self.class_info = json.load(f)
        self.augmentation = instantiate(cfg.dataloader.test.mapper.augmentation)
        
        det = instantiate(cfg.model).to('cuda')
        det.eval()
        DetectionCheckpointer(det).load(det_checkpoint)
        
        self.model = det  # 여기서는 분류기(clf) 없이 탐지기(det)만 로드

    def __call__(self, image, threshold=0.3):
        np_image = np.asarray(image)
        dataset_dict = {}
        utils.check_image_size(dataset_dict, np_image)
        np_image, _ = T.apply_transform_gens(self.augmentation, np_image)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(np_image.transpose(2, 0, 1)))
        dataset_dict["pil_image"] = image

        with torch.no_grad():
            output = self.model([dataset_dict])[0]['instances']

        output = output.to('cpu')

      
        # 클래스 정보를 무시하고 bbox 정보만 생성
        boxes = output.pred_boxes.tensor.tolist()
        scores = output.scores.tolist()
        # preds = output.pred_classes.tolist()
        preds = [0] * len(boxes)  # 모든 클래스를 bbox로 설정

        return boxes, scores, preds