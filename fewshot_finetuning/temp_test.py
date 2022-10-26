from calendar import c
import cv2
import os
import numpy as np
import PIL
import random
import json
from matplotlib import pyplot as plt
import xml.etree.ElementTree as ET
import torch
from torch import nn

basepath = '/home/dipu/deepdiscover/'
max_iter = 300
output_dir = f'./output_all_maxiter_{max_iter}'


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


def get_paisley_dicts(basepath= basepath, split='train'):
    imgfile =  f'{basepath}/codes/detection/dataset/ETHZShapeClasses-V1.2/{split}_imgs.txt'
    imfns = open(imgfile, 'r').readlines()
    imfns = [x.strip() for x in imfns]

    annfile =  f'{basepath}/codes/detection/dataset/ETHZShapeClasses-V1.2/{split}_anns.txt'
    annfns = open(annfile, 'r').readlines()
    annfns = [x.strip() for x in annfns]

    dataset_dicts = []
    for ii, (im_path, ann_path) in enumerate(zip(imfns, annfns)):
        record = {}
        fn = im_path
        filename = os.path.join(basepath, f'{im_path}')
        #print(split, filename)
        height, width = cv2.imread(filename).shape[:2]        
        record["file_name"] = filename
        record["image_id"] = fn
        record["height"] = height
        record["width"] = width
        
        # Read bbox annotatiopns
        ann_path, class_id = ann_path.split('\t')
        class_id = int(class_id)
        ann_file = os.path.join(basepath, ann_path)
        
        if ann_file.split('.')[-1] == 'xml':
        
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
                    "category_id": class_id
                }
                objs.append(obj)

        else:
            with open(ann_file) as f:
                bboxs = f.readlines()
            bboxs = [x.strip() for x in bboxs]
            
            objs = []
            for bb in bboxs:
                bb = bb.split()
                bb= [np.float32(x) for x in bb]
                obj = {
                    "bbox": bb,
                    "category_id": class_id
                }
                objs.append(obj)
            
        record["annotations"] = objs
        dataset_dicts.append(record)
    
    return dataset_dicts


data_dicts = get_paisley_dicts(basepath, split='train')