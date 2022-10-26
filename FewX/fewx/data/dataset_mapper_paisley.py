# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

import pandas as pd
from detectron2.data.catalog import MetadataCatalog

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithSupport"]


class DatasetMapperWithSupport:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

        self.tfm_gens = self.build_augmentation(cfg, is_train)
        self.apply_support_transform = cfg.INPUT.AUGMENTATIONS.APPLY_ON_SUPPORT

        if self.apply_support_transform:
            self.support_augs = self.build_support_augmentation(cfg, is_train)
        
        
        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS

        self.few_shot       = cfg.INPUT.FS.FEW_SHOT
        self.support_way       = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot       = cfg.INPUT.FS.SUPPORT_SHOT
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.proposal_min_box_size = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

        if self.is_train:
            # support_df
            self.support_on = True
            if self.few_shot:
                self.support_df = pd.read_pickle("./datasets/paisley/10_shot_support_df.pkl")
            else:
                raise('No base training for this dataset')
                #self.support_df = pd.read_pickle("./datasets/coco/train_support_df.pkl")

            metadata = MetadataCatalog.get('paisley_train')
            # #unmap the category mapping ids for COCO
            # reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
            # self.support_df['category_id'] = self.support_df['category_id'].map(reverse_id_mapper)


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        if self.is_train:
            # support
            if self.support_on:
                if "annotations" in dataset_dict:
                    # USER: Modify this if you want to keep them for some reason.
                    for anno in dataset_dict["annotations"]:
                        if not self.mask_on:
                            anno.pop("segmentation", None)
                        if not self.keypoint_on:
                            anno.pop("keypoints", None)
                support_images, support_bboxes, support_cls = self.generate_support(dataset_dict)
                dataset_dict['support_images'] = torch.as_tensor(np.ascontiguousarray(support_images))
                dataset_dict['support_bboxes'] = support_bboxes
                dataset_dict['support_cls'] = support_cls

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                self.proposal_min_box_size,
                self.proposal_topk,
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)
            
            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

    def generate_support(self, dataset_dict):
        support_way = self.support_way #2 # one for paisley
        support_shot = self.support_shot #10 during base training and 9 during finetuning
        
        id = dataset_dict['annotations'][0]['id']
        query_cls = self.support_df.loc[self.support_df['id']==id, 'category_id'].tolist()[0] # they share the same category_id and image_id
        query_img = self.support_df.loc[self.support_df['id']==id, 'image_id'].tolist()[0]
        all_cls = self.support_df.loc[self.support_df['image_id']==query_img, 'category_id'].tolist()

        # Crop support data and get new support box in the support data
        support_data_all = np.zeros((support_way * support_shot, 3, 320, 320), dtype = np.float32)
        support_box_all = np.zeros((support_way * support_shot, 4), dtype = np.float32)
        used_image_id = [query_img]

        used_id_ls = []
        for item in dataset_dict['annotations']:
            used_id_ls.append(item['id'])
        #used_category_id = [query_cls]
        used_category_id = list(set(all_cls))
        support_category_id = []
        mixup_i = 0

        for shot in range(support_shot):
            # Support image and box
            support_id = self.support_df.loc[(self.support_df['category_id'] == query_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls)), 'id'].sample(random_state=id).tolist()[0]
            support_cls = self.support_df.loc[self.support_df['id'] == support_id, 'category_id'].tolist()[0]
            support_img = self.support_df.loc[self.support_df['id'] == support_id, 'image_id'].tolist()[0]
            used_id_ls.append(support_id) 
            used_image_id.append(support_img)

            support_db = self.support_df.loc[self.support_df['id'] == support_id, :]
            assert support_db['id'].values[0] == support_id
            
            support_data = utils.read_image(support_db["file_path"].tolist()[0], format=self.img_format)
            support_box = support_db['support_box'].tolist()[0]
            if self.apply_support_transform:
                support_data, support_transforms = T.apply_transform_gens(self.support_augs, support_data)
                # assumming support has XYXYAbBS 
                support_im_size = support_data.shape[:2]
                support_box = support_transforms.apply_box(np.array([support_box]))[0].clip(min=0)
                support_box = np.minimum(support_box, list(support_im_size + support_im_size)[::-1])            
                
            support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
            
            #print(support_data)
            support_data_all[mixup_i] = support_data
            support_box_all[mixup_i] = support_box
            support_category_id.append(0) #support_cls)
            mixup_i += 1

        assert(support_way ==1)  # only one class available for paisley
        
        if support_way == 1:
            pass
        else:
            # support way is 1 for paisley as we donot have negative class atm.
            # hence the following will be skipped, 
            # do the necessary modifications in to the RCNN backbone code 
            
            for way in range(support_way-1):
                other_cls = self.support_df.loc[(~self.support_df['category_id'].isin(used_category_id)), 'category_id'].drop_duplicates().sample(random_state=id).tolist()[0]
                used_category_id.append(other_cls)
                for shot in range(support_shot):
                    # Support image and box

                    support_id = self.support_df.loc[(self.support_df['category_id'] == other_cls) & (~self.support_df['image_id'].isin(used_image_id)) & (~self.support_df['id'].isin(used_id_ls)), 'id'].sample(random_state=id).tolist()[0]
                     
                    support_cls = self.support_df.loc[self.support_df['id'] == support_id, 'category_id'].tolist()[0]
                    support_img = self.support_df.loc[self.support_df['id'] == support_id, 'image_id'].tolist()[0]

                    used_id_ls.append(support_id) 
                    used_image_id.append(support_img)

                    support_db = self.support_df.loc[self.support_df['id'] == support_id, :]
                    assert support_db['id'].values[0] == support_id

                    support_data = utils.read_image('./datasets/coco/' + support_db["file_path"].tolist()[0], format=self.img_format)
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_box = support_db['support_box'].tolist()[0]
                    support_data_all[mixup_i] = support_data
                    support_box_all[mixup_i] = support_box
                    support_category_id.append(1) #support_cls)
                    mixup_i += 1
        
        return support_data_all, support_box_all, support_category_id   # 20x3x320x320 , 20x4 ,   20 
    
    
    def build_augmentation(self, cfg, is_train):
        # TODO 
        # 1. Finegrained choice for all the augmentations and their hyperparams
        # 2. Add other augmentations that are not supported in detectron2, may be using other libraries like albumentations
        
        """
        Create a list of default :class:`Augmentation` from config.
        Now it includes resizing and flipping.

        Returns:
            list[Augmentation]
        """
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
            
        if cfg.INPUT.AUGMENTATIONS.TYPE == 'Default':
            augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
            if is_train and cfg.INPUT.RANDOM_FLIP != "none":
                augmentation.append(
                    T.RandomFlip(
                        horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                        vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                    )
                )
            return augmentation
        
        elif cfg.INPUT.AUGMENTATIONS.TYPE == 'More':
            
            if is_train:
                if cfg.INPUT.AUGMENTATIONS.ROTATION_UPPER == 0:
                    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style) ]
                else:
                    augmentation = [T.RandomRotation(angle=[0,cfg.INPUT.AUGMENTATIONS.ROTATION_UPPER]), 
                                T.ResizeShortestEdge(min_size, max_size, sample_style) ]
                    
                
                augmentation.extend([T.RandomBrightness(0.1, 2),
                                    T.RandomContrast(0.1, 4),
                                    T.RandomSaturation(0.1, 4),
                                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                                    T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
                            ])
                
            else:
                augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
                
            return augmentation
        
        else:
            raise(f'Augmentation choice {cfg.INPUT.AUGMENTATIONS} is not implemented!')
            
        
    

    def build_support_augmentation(self, cfg, is_train):
        
        if is_train:
            if cfg.INPUT.AUGMENTATIONS.ROTATION_UPPER == 0:
                augmentation = [T.RandomBrightness(0.1, 2),
                                T.RandomContrast(0.1, 4),
                                T.RandomSaturation(0.1, 4),
                                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                                T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
                                ]
            else:
                augmentation = [T.RandomRotation(angle=[0,25]),
                                T.Resize((320,320)),
                                T.RandomBrightness(0.1, 2),
                                T.RandomContrast(0.1, 4),
                                T.RandomSaturation(0.1, 4),
                                T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                                T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
                                ]
        else:
            augmentation = [T.NoOpTransform()]
            
        return augmentation
            