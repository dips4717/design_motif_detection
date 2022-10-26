import cv2
import argparse
import os
import numpy as np
import xml.etree.ElementTree as ET


import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

parser = argparse.ArgumentParser(description='Evaluation arguments for detection')
basepath = '/home/dipu/deepdiscover'

def get_paisley_dicts(img_dir= f'{basepath}/images/Paisley/', split='train'):
    txtfile = f'Paisley_{split}.txt'
    imfns = open(txtfile, 'r').readlines()
    imfns = [x.strip() for x in imfns ]
    
    dataset_dicts = []
    for ii, im_path in enumerate(imfns):
        record = {}
        fn = im_path
        filename = os.path.join(img_dir, f'{im_path}.png')
        #print(split, filename)
        height, width = cv2.imread(filename).shape[:2]        
        record["file_name"] = filename
        record["image_id"] = fn
        record["height"] = height
        record["width"] = width
        # Read bbox annotatiopns
        ann_file = os.path.join(img_dir, f'{im_path}.xml')
        tree = ET.parse(ann_file)
        root = tree.getroot()
        objs = []
        
        for boxes in root.iter('object'):

            ymin, xmin, ymax, xmax = None, None, None, None

            ymin = int(boxes.find("bndbox/ymin").text)
            xmin = int(boxes.find("bndbox/xmin").text)
            ymax = int(boxes.find("bndbox/ymax").text)
            xmax = int(boxes.find("bndbox/xmax").text)
            bb = [xmin, ymin, xmax, ymax]
            bb= [np.float32(x) for x in bb]
            obj = {
                "bbox": bb,
                "category_id": 0,
                "bbox_mode": BoxMode.XYXY_ABS
            }
            objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


for d in ["train", "test"]:
    DatasetCatalog.register("paisley_" + d, lambda d=d: get_paisley_dicts(f'{basepath}/images/Paisley/', d))
    MetadataCatalog.get("paisley_" + d).set(thing_classes=["Paisley"])

paisley_metadata = MetadataCatalog.get("paisley_train")
output_dir = 'output_paisley_maxiter_300/'

for model_name in ['model_final.pth', 
                    'model_0000099.pth', 'model_0000199.pth',
                    'model_0000299.pth']: # 'model_0000399.pth',
                    #'model_0000499.pth',  'model_0000599.pth' ]:
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("paisley_train",)
    cfg.DATASETS.TEST = ("paisley_test",)

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST =  0.5
    cfg.MODEL.WEIGHTS = output_dir + model_name # "model_final.pth"


    # trainer = DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=False)
    # evaluator = COCOEvaluator("paisley_test", cfg, False, output_dir=output_dir)
    # val_loader = build_detection_test_loader(cfg, "paisley_test")
    # inference_on_dataset(trainer.model, val_loader, evaluator)
    # another equivalent way is to use trainer.test

    evaluator = COCOEvaluator("paisley_test", cfg, False, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, "paisley_test")
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    inference_on_dataset(model, val_loader, evaluator)
