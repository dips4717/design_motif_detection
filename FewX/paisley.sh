# rm support_dir/support_feature_paisley.pkl
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
#     MODEL.BACKBONE.FREEZE_AT 3  OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat3 2>&1 | tee log/paisely_fsod_finetune_train_log_freeze3.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat3/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_freeze3.txt


# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat3/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_freeze3.txt


# rm support_dir/support_feature_paisley.pkl
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
#     MODEL.BACKBONE.FREEZE_AT 1  OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat1 2>&1 | tee log/paisely_fsod_finetune_train_log_freeze1.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat1/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_freeze1.txt


# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat1/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_freeze1.txt


# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
# 	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
# 	--eval-only MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x/model_final_gpu1_ims4.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_freeze5.txt


rm support_dir/support_feature_paisley.pkl
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
    MODEL.BACKBONE.FREEZE_AT 2  OUTPUT_DIR ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat2 2>&1 | tee log/paisely_fsod_finetune_train_log_freeze2.txt

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
	--eval-only MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_freeze2.txt


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train_debug_paisley.py --num-gpus 4 \
	--config-file configs/fsod/finetune_R_50_C4_1x_paisley.yaml \
	--eval-only MODEL.WEIGHTS ./output/fsod/paisley_finetune_dir/R_50_C4_1x_freezeat2/model_final.pth 2>&1 | tee log/paisley_fsod_finetune_test_log_freeze2.txt


Visualize the proposals  --VIS_PERIOD
Get the high confident proposals.
