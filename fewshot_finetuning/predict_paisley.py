import cv2
import numpy as np
import os
import PIL 
from matplotlib import pyplot as plt

import xml.etree.ElementTree as ET
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.modeling import build_model
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
dataset_dicts = get_paisley_dicts(split = 'test')

output_dir = 'output_all_maxiter_300/'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("paisley_train",)
cfg.DATASETS.TEST = ("paisley_test",)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2 # 0.5
cfg.MODEL.WEIGHTS = output_dir + "model_final.pth"
predictor = DefaultPredictor(cfg)


test_images = dataset_dicts
save_dir = 'Predicted/Paisley/test_thr0.2/'
os.makedirs(save_dir, exist_ok=True)

for ii, d in enumerate(test_images):
    im = cv2.imread(d["file_name"])    
    fn = d["image_id"]
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=paisley_metadata, scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = v.get_image()[:, :, ::-1]
    img = my_imshow(img)
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.setp(ax,  xticklabels=[], yticklabels=[])
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(0,0,1,1,0,0)
    ax.margins(0,0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_rasterization_zorder(1) 
    plt.savefig( save_dir+fn+'.png', bbox_inches='tight', pad_inches = 0, dpi = 300)
    plt.close()
    
    