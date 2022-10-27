#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Debug code for FewX training code

This script is a simplified version of the training script in detectron2/tools.
"""

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import cv2
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
#from detectron2.evaluation import COCOEvaluator
#from detectron2.data import build_detection_train_loader
from detectron2.data import build_batch_data_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.engine import DefaultPredictor
from fewx.config import get_cfg
from fewx.data.dataset_mapper_paisley import DatasetMapperWithSupport
from fewx.data.build import build_detection_train_loader, build_detection_test_loader
from fewx.solver import build_optimizer
from fewx.evaluation import COCOEvaluator
import PIL

import bisect
import copy
import itertools
import logging
import numpy as np
import operator
import pickle
import torch.utils.data

import detectron2.utils.comm as comm
from detectron2.utils.logger import setup_logger

from matplotlib import pyplot as plt


class Trainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithSupport(cfg)
        return build_detection_train_loader(cfg, mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name)

    @classmethod
    def build_optimizer(cls, cfg, model):
        """
        Returns:
            torch.optim.Optimizer:
        It now calls :func:`detectron2.solver.build_optimizer`.
        Overwrite it if you'd like a different optimizer.
        """
        return build_optimizer(cfg, model)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

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



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="fewx")

    return cfg


def main(args):
    cfg = setup(args)
    predictor = DefaultPredictor(cfg)
    
    def load_paisley_dict(split):
        with open(f'datasets/paisley/paisley_{split}_dicts.pkl', 'rb') as f:
            dicts = pickle.load(f)
        return dicts
     
    
    for d in ["train", "test"]: 
        if f'paisley_{d}' not  in DatasetCatalog.keys():
            DatasetCatalog.register("paisley_" + d, lambda d=d: load_paisley_dict(d))
            MetadataCatalog.get("paisley_" + d).set(thing_classes=["Paisley"])

    paisley_metadata = MetadataCatalog.get("paisley_train")
    
    save_dir = f'{args.model_path}/Predicted/test_thr0.2/'
    os.makedirs(save_dir, exist_ok=True)
    test_dict = load_paisley_dict('test')
    
    for ii, d in enumerate(test_dict):
        im = cv2.imread(d["file_name"])    
        fn = d["image_id"]
        print(fn)
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



if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus = 1
    #print(vars(args))
    
    args.model_path = './output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0'
    
    args.config_file  = f'{args.model_path}/config.yaml'
    args.opts = ['MODEL.WEIGHTS', f'{args.model_path}/model_final.pth']
    args.eval_only = True
    
    print("Command Line Args:", args)
    main(args)


# if __name__ == "__main__":
#     args = default_argument_parser().parse_args()
#     print("Command Line Args:", args)
#     launch(
#         main,
#         args.num_gpus,
#         num_machines=args.num_machines,
#         machine_rank=args.machine_rank,
#         dist_url=args.dist_url,
#         args=(args,),
#     )
    

