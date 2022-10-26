import cv2
import numpy as np
import os
import PIL 
from matplotlib import pyplot as plt

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
dataset_dicts = get_applelogo_dicts(split = 'train')

output_dir = 'output/'
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("applelogos_train",)
cfg.DATASETS.TEST = ("applelogos_test",)

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 #128   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = output_dir + "model_final.pth"
predictor = DefaultPredictor(cfg)


test_images = dataset_dicts
save_dir = 'Predicted/Applelogos/train/'
os.makedirs(save_dir, exist_ok=True)

for ii, d in enumerate(test_images):
    im = cv2.imread(d["file_name"])    
    fn = d["image_id"]
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=applelogos_metadata, scale=1)
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
    
    