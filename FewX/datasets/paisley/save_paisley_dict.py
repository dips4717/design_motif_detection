import os
import numpy as np
from detectron2.structures import BoxMode
import xml.etree.ElementTree as ET
import pickle
import cv2


def get_paisley_dicts(img_dir= f'/home/dipu/deepdiscover/images/Paisley/', split='train'):
    txtfile = f'../../../fewshot_finetuning/Paisley_{split}.txt'
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

if __name__ == "__main__":
    basepath = '/home/dipu/deepdiscover'
    for split in ['train', 'test', 'test_v2']:
        #split = 'test_v2' # train/test/test_v2
        dataset_dict = get_paisley_dicts(split=split)
        print('Number of images in the dataset: ', len(dataset_dict))
        
        # Save this to reduce overhead during training/test
        # with open(f'paisley_{split}_dicts.pkl', 'wb') as f:
        #     pickle.dump(dataset_dict, f)