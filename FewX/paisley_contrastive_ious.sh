rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
    --config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.7 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.7 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.7.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.7/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.7/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.7.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.7/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.7/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.7.txt

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.8 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.8 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.8.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.8/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.8/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.8.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.8/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.8/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.8.txt

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.9 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.9.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.9.txt

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD 0.95 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.95 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2_iou0.95.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.95/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.95/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.95.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.95/config.yaml \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.95/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2_iou0.95.txt