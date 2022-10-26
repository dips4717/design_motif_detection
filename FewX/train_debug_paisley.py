#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Debug code for FewX training code

This script is a simplified version of the training script in detectron2/tools.
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
#from detectron2.evaluation import COCOEvaluator
#from detectron2.data import build_detection_train_loader
from detectron2.data import build_batch_data_loader
from detectron2.data import DatasetCatalog, MetadataCatalog


from fewx.config import get_cfg
from fewx.data.dataset_mapper_paisley import DatasetMapperWithSupport
from fewx.data.build import build_detection_train_loader, build_detection_test_loader
from fewx.solver import build_optimizer
from fewx.evaluation import COCOEvaluator

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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    #cfg.MODEL.BACKBONE.FREEZE_AT = 1
    #cfg.DATALOADER.NUM_WORKERS = 0
    #cfg.SOLVER.IMS_PER_BATCH = 8
    #cfg.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS = True
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="fewx")

    return cfg


def main(args):
    cfg = setup(args)
    
    def load_paisley_dict(split):
        with open(f'datasets/paisley/paisley_{split}_dicts.pkl', 'rb') as f:
            dicts = pickle.load(f)
        return dicts
    
    for d in ["train", "test", "test_v2"]: 
        if f'paisley_{d}' not  in DatasetCatalog.keys():
            DatasetCatalog.register("paisley_" + d, lambda d=d: load_paisley_dict(d))
            MetadataCatalog.get("paisley_" + d).set(thing_classes=["Paisley"])

    

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()



# ## Debug mode. 
# if __name__ == "__main__":
#     args = default_argument_parser().parse_args()
#     args.num_gpus = 1
#     #print(vars(args))
    
#     #Train
#     args.config_file  = 'configs/fsod/finetune_R_50_C4_1x_paisley.yaml'
#     # model weights initialized using trained weight for debuging behaviour towards end of training, not necessarily required.
#     args.opts = ['MODEL.WEIGHTS', './output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat2/model_final.pth', 
#                   'MODEL.BACKBONE.FREEZE_AT', 2,
#                   'DATALOADER.NUM_WORKERS',0,
#                   'MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS', True,
#                   'SOLVER.IMS_PER_BATCH', 4,
#                   'INPUT.AUGMENTATIONS.TYPE', 'More',
#                   'INPUT.AUGMENTATIONS.APPLY_ON_SUPPORT', True,
#                   'INPUT.AUGMENTATIONS.ROTATION_UPPER', 0
#                   ]    
        
#     # ## Eval
#     # args.opts = ['MODEL.WEIGHTS', './output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat2/model_final.pth', 
#     #             'MODEL.BACKBONE.FREEZE_AT', 2]
#     # args.config_file  = './output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat2/config.yaml'
#     # args.eval_only = True
        
    
#     print("Command Line Args:", args)
#     main(args)

## Training 
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print(vars(args))
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    

