## Dataset
- Download 
- Dataset splits
  - train 10 images
  - test 10 images
  - test_v2 images [Excludes a image with lots of small bboxes.]

## Data Preparation
- Generate dataset dictionaries
  - Just to precompute dataset annotation dictionaries but can be done online during training.
  - Run `datasets/paisley/save_paisley_dict.py`. This reads the list of images in each split set provided in [fewshot_finetuning folder] (../fewshot_finetuning) 

