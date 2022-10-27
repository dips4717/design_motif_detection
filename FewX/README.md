# Few-shot Object Detection based on [FewX](https://github.com/fanq15/FewX)
This contains implementation of few-shot object detection adapted from on original FewX repo. 
- For now, This repo only uses positive support branch
- We implement the contrastive loss into the codebase see config files
- We experiment with different sets of data augmentations


## Dataset
- Download 
- Dataset splits
  - train 10 images
  - test 10 images
  - test_v2 images [Excludes a image with lots of small bboxes.]

## Data Preparation
- **Generate dataset dictionaries**
  - Just to precompute dataset annotation dictionaries but can be done online during training.
  - Run `datasets/paisley/save_paisley_dict.py` to generate pkl files. This reads the list of images in each split set (txt files) provided in [fewshot_finetuning folder](../fewshot_finetuning) 
  
- **Generate Support images**
  - Run `datasets/paisley/ge_support_pool_fewshot.py`. Change the paths prior execution.
  - This will generate support images (cropped instances) and save them into `10image_shot_support`, and related information into a dataframe `10_shot_support_df.pkl`

- **Pretrained Base Model**
  - Download a pre-trained base model provided by [original repo](https://github.com/fanq15/FewX)  [base_model](https://drive.google.com/file/d/1VdGVmcufa2JBmZUfwAcDj1OL5tKTFhQ1/view) into  `./output/fsod/R_50_C4_1x/model_final.pth`.
  - Used for initializing the model
  - The model architecture is Res50-C4 

## Training and Evaluation

**Training**
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.9 \
    INPUT.AUGMENTATIONS.TYPE More \
    INPUT.AUGMENTATIONS.APPLY_ON_SUPPORT False \
    INPUT.AUGMENTATIONS.ROTATION_UPPER 25 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25.txt
```

The training can be carried out using above code based on the config file. See `configs/fsod/finetune_R_50_C4_1x_paisley.yaml` for 
The config parameter can be changed in command-line as above.

**Evaluate**
```
rm -rf support_dir/support_feature_paisley.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25.txt
```

To evaluate, run the commands above pointing the config file and saved model (outputs of training)
Delete the support features first if this exists
The evaluation command is run twice, the first run will compute the support features and the second run will actually compute the metrics.

** Evaluation on test set v2 **
To run evaluation on test_v2, just change the test set argument as below.

rm -rf support_dir/support_feature_paisley.pkl

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/config.yaml \
	--eval-only \
  DATASETS.TEST '("paisley_test_v2",)' \
  MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/config.yaml \
	--eval-only \
  DATASETS.TEST '("paisley_test_v2",)' \
  MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25.txt
```


## Visualize Predictions
To visualize/plot the detection results on images, simply run `predict.py` with necessary arguments for model_path and config.


## Augmentations
Currently, only augmentations in detectron2 is used.
- `Default` and `More` data augmentation configs
- `ROTATION_UPPER` for upper bound of random rotation, 0 means no rotation augmentation
- You can play with more config parameters, see config files, add new config parameters, 
See `fewx/data/dataset_mapper_paisley.py` for actual implementation
TODO:
  - To integrate libraries like Albumentations into detectron2 or writing up the custom augmentation in detectron2 would be need.

## Note: To add more config parameters.
- Add the parameters you want to add into `fewx/config/defaults.py` first.
- This can be then reffered in config.yaml files and also given as command-line args.


## Implementation Notes

## Acknowlegement
[Detectron2](https://github.com/facebookresearch/detectron2)
[FewX](https://github.com/fanq15/FewX)
[FSCE](https://github.com/megvii-research/FSCE)