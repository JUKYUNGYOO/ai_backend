import os
import json
import logging
import requests
import logging.config
from datetime import datetime
from collections import defaultdict, Counter

#---------------------------------------------------------------------------
#  Logging
#---------------------------------------------------------------------------

logging.config.dictConfig({
    "version": 1,
    "formatters": {
        "json": {
            "format" : '{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}',
        }
    },
    "handlers": {
        "stdout": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "json",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "orion": {
            "level": "INFO",
            "handlers": ["stdout"]
        }
    }
})

class OrionLogger:

    def __init__(self):
        self._logger = logging.getLogger("orion")

    def info(self, message):
        self._logger.info(json.dumps(message, ensure_ascii=False))

    def error(self, message, send_to_kakaowork=False):
        self._logger.error(json.dumps(message, ensure_ascii=False))

#---------------------------------------------------------------------------
#   Upload & Download failed images
#---------------------------------------------------------------------------

class ErrorFiles:

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def save(self, filename, image_content):
        saved_path = os.path.join(self.root, f"{datetime.now().strftime('%Y%m%d_%H%M')}_{filename}")
        with open(saved_path, "wb+") as f:
            f.write(image_content)
        return saved_path


class OrionDataManager():

    def __init__(self, boxes, scores, preds, class_info, image, image_filename):
        self.boxes = boxes
        self.scores = scores
        self.preds = preds
        self.class_info = class_info
        self.image = image
        self.image_filename = image_filename

    def image_info_generate(self):
        # Image_info
        image_width, image_height = self.image.size
        self.image_info = {
                        'filename' : self.image_filename , 
                        'image_width' : image_width, 
                        'image_height' : image_height
                        }

    def object_generate(self):
        # Objects
        self.objects = []
        for (x1, y1, w, h), conf, pred in zip(self.boxes, self.scores, self.preds):
            class_info = self.class_info[pred]
            self.objects.append({
                "name": class_info["name"],
                "manufacturer": class_info["manufacturer"],
                "code": class_info["id"],
                "bbox": [x1, y1, w, h],
                "confidence": conf
            })

    def area_generate(self):
        # Area 
        self.tot_area = 0
        self.product_area = defaultdict(float)
        self.manufacturer_ctrs = defaultdict(list)
        self.manufacturer_area = defaultdict(float)
        for o in self.objects:
            _, _, w, h = o["bbox"]
            self.tot_area += w*h
            self.product_area[o["name"]] += w*h
            self.manufacturer_ctrs[o["manufacturer"]].append(o["name"])
            self.manufacturer_area[o["manufacturer"]] += w*h
        self.area = {"total_area" : self.tot_area, "manufacutrer_area" : self.manufacturer_area, "product_area" : self.product_area}

    def products_generate(self):
        # Products
        self.products = []
        unkown_class = []

        product_ctrs = Counter(self.preds)
        for k in product_ctrs:
            class_info = self.class_info[k]
            if class_info["name"] == "알수없음":
                unkown_class.append({
                    "name": class_info["name"],
                    "manufacturer": class_info["manufacturer"],
                    "code": class_info["id"],
                    "counts": product_ctrs[k],
                    "total_proportion" : self.product_area[class_info["name"]] / self.tot_area
                })
            else:
                self.products.append({
                    "name": class_info["name"],
                    "manufacturer": class_info["manufacturer"],
                    "code": class_info["id"],
                    "counts": product_ctrs[k],
                    "total_proportion" : self.product_area[class_info["name"]] / self.tot_area
                })
        self.products.sort(key=lambda p: p["total_proportion"], reverse=True)
        self.products += unkown_class

    def manufacturers_generate(self):
        # Manufacturers
        self.manufacturers = [] 
        unkown_class = []

        for manufacturer_name, objs in self.manufacturer_ctrs.items():
            if manufacturer_name == "기타":
                unkown_class.append({
                    manufacturer_name: {
                        "products": len(set(objs)),
                        "productCounts": len(objs),
                        "manufacturer_proportion": self.manufacturer_area[manufacturer_name] / self.tot_area , 
                        "products_proportion" : {product_name : self.product_area[product_name] / self.manufacturer_area[manufacturer_name] for product_name in set(objs)}
                    }
                })
            else:
                self.manufacturers.append({
                    manufacturer_name: {
                        "products": len(set(objs)),
                        "productCounts": len(objs),
                        "manufacturer_proportion": self.manufacturer_area[manufacturer_name] / self.tot_area , 
                        "products_proportion" : {product_name : self.product_area[product_name] / self.manufacturer_area[manufacturer_name] for product_name in set(objs)}
                    }
                })
        self.manufacturers.sort(key=lambda p: next(iter(p.values()))['manufacturer_proportion'], reverse=True)
        self.manufacturers += unkown_class


    def result_extract(self):
        # inference result
        self.image_info_generate()
        self.object_generate()
        self.area_generate()
        self.products_generate()
        self.manufacturers_generate()

        
        self.result = {
            "image_info" : self.image_info, 
            "objects": self.objects,
            "area" : self.area,
            "products": self.products,
            "manufacturers": self.manufacturers
        }
        return self.result
    

    def total_result_extract(self):
        # inference result
        self.object_generate()
        self.area_generate()
        self.products_generate()
        self.manufacturers_generate()

        
        self.result = {
            "area" : self.area,
            "products": self.products,
            "manufacturers": self.manufacturers
        }
        return self.result