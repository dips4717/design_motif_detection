import cv2
import os
import numpy as np
import PIL
import random
import json
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'




# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
#%%
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from torch.nn import functional as F
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.modeling import  ROI_BOX_HEAD_REGISTRY
from detectron2.structures import BoxMode

#basepath = '/vol/research/deepdiscover'
basepath = '/home/dipu/deepdiscover'

def my_imshow(a):
    a = a.clip(0, 255).astype('uint8')
      # cv2 stores colors as BGR; convert to RGB
    if a.ndim == 3:
        if a.shape[2] == 4:
          a = cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA)
        else:
          a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    a = PIL.Image.fromarray(a)
    return(a)


def get_applelogo_dicts(img_dir= f'{basepath}/codes/detection/dataset/ETHZShapeClasses-V1.2/', split='train'):
    txtfile = img_dir + f'Applelogos_{split}.txt'
    imfns = open(txtfile, 'r').readlines()
    imfns = [x.strip() for x in imfns ]
    
    dataset_dicts = []
    for ii, im_path in enumerate(imfns):
        record = {}
        fn = im_path.split('/')[-1].split('.')[0]
        filename = os.path.join(img_dir, im_path)
        height, width = cv2.imread(filename).shape[:2]        
        record["file_name"] = filename
        record["image_id"] = fn
        record["height"] = height
        record["width"] = width
        # Read bbox annotatiopns
        ann_file = img_dir + f'Applelogos/{fn}_applelogos.groundtruth'
        with open(ann_file) as f:
            bboxs = f.readlines()
        bboxs = [x.strip() for x in bboxs]
        
        objs = []
        for bb in bboxs:
            bb = bb.split()
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
    DatasetCatalog.register("applelogos_" + d, lambda d=d: get_applelogo_dicts(f'{basepath}/codes/detection/dataset/ETHZShapeClasses-V1.2/', d))
    MetadataCatalog.get("applelogos_" + d).set(thing_classes=["applelogo"])

applelogos_metadata = MetadataCatalog.get("applelogos_train")

"""To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:"""
dataset_dicts = get_applelogo_dicts(f'{basepath}/codes/detection/dataset/ETHZShapeClasses-V1.2/', 'test')
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=applelogos_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    img = my_imshow(vis.get_image()[:, :, ::-1])
      
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plt.setp(ax,  xticklabels=[], yticklabels=[])
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    
# Fine tune 
"PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("applelogos_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon)
#cfg.MODEL.ROI_BOX_HEAD.NAME = 'FastRCNNConvFCHead_V2'
#cfg.HEAD_TYPE = 'comat'
#cfg.MODEL.ROI_HEADS.NAME = 'StandardROIHeads_Graph'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()
    

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("applelogos_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "applelogos_test")
inference_on_dataset(trainer.model, val_loader, evaluator)
# another equivalent way is to use trainer.test


