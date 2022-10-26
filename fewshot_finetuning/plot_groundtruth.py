import cv2
import os
import numpy as np
import PIL
import random
import json
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from torch.nn import functional as F
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.modeling import  ROI_BOX_HEAD_REGISTRY
from detectron2.structures import BoxMode

#basepath = '/vol/research/deepdiscover'
basepath = '/home/dipu/deepdiscover'
max_iter = 300
output_dir = f'./output_paisley_maxiter_{max_iter}_C4'

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
    if f'paisley_{d}' not  in DatasetCatalog.keys():
        DatasetCatalog.register("paisley_" + d, lambda d=d: get_paisley_dicts(f'{basepath}/images/Paisley/', d))
        MetadataCatalog.get("paisley_" + d).set(thing_classes=["Paisley"])

paisley_metadata = MetadataCatalog.get("paisley_train")

"""To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:"""


def plot(split):
    dataset_dicts = get_paisley_dicts(f'{basepath}/images/Paisley/', split)
    for ii,  d in enumerate(dataset_dicts):
        img = cv2.imread(d["file_name"])
        fn = d["image_id"]
        visualizer = Visualizer(img[:, :, ::-1], metadata=paisley_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        img = my_imshow(vis.get_image()[:, :, ::-1])
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        plt.setp(ax,  xticklabels=[], yticklabels=[])
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.tight_layout()
        plt.subplots_adjust(0,0,1,1,0,0)
        ax.margins(0,0)
        ax.xaxis.set_major_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_rasterization_zorder(1) 
        savefolder = f'Groudtruth_Plots/{split}'
        os.makedirs(savefolder, exist_ok=True)
        plt.savefig(f'{savefolder}/{fn}.png', bbox_inches='tight', pad_inches = 0, dpi = 300)
        
for split in ['train', 'test']:
    plot(split)