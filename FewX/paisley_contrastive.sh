# rm support_dir/support_feature_paisley.pkl
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
#     MODEL.BACKBONE.FREEZE_AT 2  \
#     MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
#     OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon 2>&1 | tee log/paisely_fsod_finetune_train_log_freeze2.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
# 	--eval-only \
#     MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_supcon.txt
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
# 	--eval-only \
#     MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_supcon.txt
# ==================
rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
    --config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 10.0 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt10.0_freeze2 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt10.0_freeze2.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt10.0_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt10.0_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt10.0_freeze2.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt10.0_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt10.0_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt10.0_freeze2.txt



# =============

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 1.0 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1.0x_Supcon_wt1_freeze2 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt1.0_freeze2.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1.0x_Supcon_wt1_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1.0x_Supcon_wt1_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt1.0_freeze2.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1.0x_Supcon_wt1_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1.0x_Supcon_wt1_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt1.0_freeze2.txt




# ===============

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.1 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.1_freeze2.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.1_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.1_freeze2.txt


##  ===

rm -rf support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS True \
    MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT 0.01 \
    OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.01_freeze2 2>&1 | tee log/paisely_fsod_finetune_train_log_Supcon_wt0.01_freeze2.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.01_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.01_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.01_freeze2.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.01_freeze2 \
	--eval-only \
    MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_Supcon_wt0.01_freeze2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_R_50_C4_1x_Supcon_wt0.01_freeze2.txt
