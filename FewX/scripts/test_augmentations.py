from genericpath import exists
import cv2
import os
import PIL
import copy
import numpy as np
import torch
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels
import albumentations as A


import pickle
import pandas as pd





class Visualizer2 (Visualizer):
    """
    Same as Visualize
    Implements draw_instances
    """

    def __init__(self, img_rgb, metadata=None, scale=1.0,):
        super().__init__(img_rgb,metadata=metadata,scale=1.0)

    def draw_instances(self, instances):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = instances.gt_boxes if instances.has("gt_boxes") else None
        scores = instances.scores if instances.has("scores") else None
        classes = instances.gt_classes.tolist() if instances.has("gt_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None

        masks = None
        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (instances.pred_masks.any(dim=0) > 0).numpy()
                    if instances.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=1.0,
        )
        return self.output




def load_paisley_dict(split):
    with open(f'datasets/paisley/paisley_{split}_dicts.pkl', 'rb') as f:
        dicts = pickle.load(f)
    return dicts

dataset_dicts = load_paisley_dict('train')
for d in ["train", "test"]: 
    if f'paisley_{d}' not  in DatasetCatalog.keys():
        DatasetCatalog.register("paisley_" + d, lambda d=d: load_paisley_dict(d))
        MetadataCatalog.get("paisley_" + d).set(thing_classes=["Paisley"])

paisley_metadata = MetadataCatalog.get("paisley_train")
    

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

def build_augmentation(is_train):
    """
    This is default augmentation in the detectron2 
    """
    if is_train:
        min_size = (440, 472, 504, 536, 568, 600)
        max_size = 1000
        sample_style = "choice"
    else:
        min_size = 600
        max_size = 1000
        sample_style = "choice"
    
    #T.ResizeShortestEdge(min_size, max_size, sample_style)
    augmentation = [T.RandomRotation(angle=[0,25]), T.ResizeShortestEdge(min_size, max_size, sample_style) ]
    # augmentation = [A.HorizontalFlip(p=0.5)]
    # augmentation.append(
    #         T.RandomFlip(
    #             horizontal=1,
    #             vertical=0,
    #         )
    #     )
    augmentation.extend([
    T.RandomBrightness(0.1, 2),
    T.RandomContrast(0.1, 4),
    T.RandomSaturation(0.1, 4),
    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
    T.RandomFlip(prob=0.5, horizontal=False, vertical=True)
     ])

    return augmentation


save_dir = 'Augmentation_checks_albumentation/'
os.makedirs(save_dir, exist_ok=True)

augs = build_augmentation(True)
recompute_boxes= False


for jj in range(3):
    for ii, dataset_dict_tmp in enumerate(dataset_dicts):
        dataset_dict = copy.deepcopy(dataset_dict_tmp)  # it will be modified by code below
        fn = dataset_dict_tmp["image_id"]
        # Plot the original image and annotations
        img1 = cv2.imread(dataset_dict_tmp["file_name"])
        visualizer = Visualizer(img1[:, :, ::-1], metadata=paisley_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(dataset_dict_tmp)
        img1 = vis.get_image()[:, :, ::-1]
        print(f'Original Image Size: {img1.shape} ')
        img1 = my_imshow(img1)
            
        # PArt of code from data_mapper in detectron2
        image = utils.read_image(dataset_dict["file_name"], format='BGR') # returns image BGR numpy
        utils.check_image_size(dataset_dict, image)
        
        
        # Detectron 2 
        # aug_input = T.AugInput(image)
        # augs_list = T.AugmentationList(augs)
        # transforms = augs_list(aug_input)  # inplace 
        # image = aug_input.image
        # image_shape = image.shape[:2]
        
        # annos = [
        #         utils.transform_instance_annotations(
        #             obj, transforms, image_shape, keypoint_hflip_indices=None
        #         )
        #         for obj in dataset_dict.pop("annotations")
        #         if obj.get("iscrowd", 0) == 0
        #     ]
        
        # instances = utils.annotations_to_instances(
        #     annos, image_shape, mask_format='polygon'
        # )
        
        
        # # FewX type
        image, transforms = T.apply_transform_gens(augs, image)
        image_shape = image.shape[:2]
        annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=None
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
        
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format='polygon'
        )

        v = Visualizer2(image[:, :, ::-1], metadata=paisley_metadata, scale=1)
        v = v.draw_instances(instances)
        img = v.get_image()[:, :, ::-1]
        print(f'Transformed Image Size: {img.shape} ')
        img = my_imshow(img)

        print(transforms)
        
        with open('transforms2.txt', 'a') as f:
            f.write(f'{jj}-{ii} \t{transforms}\n')
         
        
        ## Matplotlib
        fig, ax = plt.subplots(1,2)
        plt.setp(ax,  xticklabels=[], yticklabels=[])
        ax[0].imshow(img1)
        ax[0].axis('off')
        ax[1].imshow(img)
        ax[1].axis('off') 
        
        print('Done')
        
        #plt.title(transforms)
        plt.tight_layout()
        
        #plt.show()   
        #print(dataset_dict_tmp["file_name"])
        plt.savefig(f'{save_dir}{jj}-{ii}.png', bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close()
        
        
        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)
    
