import os
import json
import torchvision.transforms as T
import torch.distributed as dist

from PIL import Image
from tqdm import tqdm
from pathlib import Path
from timm.data import ImageDataset
from concurrent.futures import ThreadPoolExecutor


class TempImageDataset(ImageDataset):

    def __init__(self, annotation, image_root, temp_dir, is_main):
        
        self.temp_dir = temp_dir
        
        with open(annotation, 'r') as f:
            anno = json.load(f)

        categories = {}
        class_map = {}
        for idx, cat in enumerate(anno['categories']):
            categories[cat['id']] = cat['name']
            class_map[cat['name']] = idx
        # is_main이 True인 경우, 주석에 따라 이미지를 자르고 카테고리 이름별로 구성된 임시 디렉토리에 저장함.
        if is_main:
            images = {}
            image_paths = {}
            for img in anno['images']:
                image_paths[img['id']] = os.path.join(image_root, img['file_name'])
                images[image_paths[img['id']]] = []

            for a in anno['annotations']:
                images[image_paths[a['image_id']]].append((categories[a['category_id']], a['bbox'], a['id']))

            assert Path(self.temp_dir).parent.stem.startswith('temp_imagefolder'), f'{Path(self.temp_dir).parent.stem} should start with "temp_imagefolder"'
            for c in categories.values():
                os.makedirs(os.path.join(self.temp_dir, c))

            with ThreadPoolExecutor(16) as executor:
                for _ in tqdm(
                    executor.map(self.crop_image, images.keys(), images.values()),
                    total=len(images),
                    desc=f'Cropping Dataset for {Path(annotation).name} at {self.temp_dir}'
                ):
                    pass
            dist.is_initialized() and dist.barrier()
        else:
            dist.is_initialized() and dist.barrier()
    
        super().__init__(self.temp_dir, class_map=class_map)
    
    def crop_image(self, image_path, bboxes):
        img = Image.open(image_path)
        for category_name, (x1, y1, w, h), bbox_id in bboxes:
            roi = img.crop((x1, y1, x1+w, y1+h))
            roi.save(os.path.join(self.temp_dir, category_name, str(bbox_id) + '.jpg'))


def create_dataset(args, is_training=False):
    
    if is_training:
        return TempImageDataset(
                    annotation='/workspace/orion_data/annotations/annotation_train.json',
                    image_root='/workspace/orion_data/images/',
                    temp_dir=os.path.join(args.data_dir, 'train'),
                    is_main=args.local_rank == 0
                )
    else:
        return TempImageDataset(
                    annotation='/workspace/orion_data/annotations/annotation_valid.json',
                    image_root='/workspace/orion_data/images/',
                    temp_dir=os.path.join(args.data_dir, 'validation'),
                    is_main=args.local_rank == 0
                )


def create_transform(is_train=False):
    
    if is_train:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        ])
    else:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250])
        ])
