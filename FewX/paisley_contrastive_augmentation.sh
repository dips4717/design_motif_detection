rm -rf support_dir/support_feature_paisley.pkl
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

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R25.txt

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.9 \
    INPUT.AUGMENTATIONS.TYPE More \
    INPUT.AUGMENTATIONS.APPLY_ON_SUPPORT False \
    INPUT.AUGMENTATIONS.ROTATION_UPPER 0 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0.txt

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.9 \
    INPUT.AUGMENTATIONS.TYPE More \
    INPUT.AUGMENTATIONS.APPLY_ON_SUPPORT True \
    INPUT.AUGMENTATIONS.ROTATION_UPPER 25 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R25.txt


rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.9 \
    INPUT.AUGMENTATIONS.TYPE More \
    INPUT.AUGMENTATIONS.APPLY_ON_SUPPORT True \
    INPUT.AUGMENTATIONS.ROTATION_UPPER 0 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA1_R0.txt


rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0/config.yaml \
	--eval-only \
    DATASETS.TEST '("paisley_test_v2",)' \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9_aug_SA0_R0_testv2.txt
