# Finetuning Faster-RCNN on Paisley dataset

## Finetune/train and evaluate 
**Training**
Run `python finetune_paisley.py`
  - Notes:
      - This will train and evaluate on paisley dataset print out the detection performance.
      - Make sure to ammend data path. This could be modifying `img_dir` to point to the folder you downloaded  
      - This code is inspired from detecton2 [tutorial] (https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5)
      - You can change the model by changing `model_config_name` and use any of yaml file from [configs](https://github.com/facebookresearch/detectron2/tree/main/configs/COCO-Detection)
      - I evaluated with models:  `faster_rcnn_R_50_FPN_3x.yaml` and `"COCO-Detection/faster_rcnn_R_50_C4_3x.yaml`
      
**Evaluation**      
Run `evaluate.py`
  - Trained model is reffered as  `cfg.MODEL.WEIGHTS` in the code, change this as appropriate.

**Visualize**

Run `predict_paisley.py`

## Things Tried
- Motivated by learning appearance-variant features, I tried to finetune together with instances of [ETHZ shape classes dataset] (http://calvin-vision.net/datasets/ethz-shape-classes/)
- Unfortunately, this did not improve performance on individual classes
- If you want to explore that further, See `fine_tune_all.py`. You will need to download ETHZ dataset and make necessasry changes into path in the code.
 