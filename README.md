  # Design Motif Detection
<div  align="center">
<img  src="paisleys.png"/>
</div>
  

This project aims at building model for detect/recognise design motifs under few-shot settings. This repo includes two approaches to do so
 

1. Finetuning Faster-RCNN. This is a simple finetuning of Faster-RCNN based architectures such as FPN and R_C4 etc. See [fewshot_finetuning](fewshot_finetuning/) for details.
2. Few-shot object detection based on |[FewX](https://github.com/fanq15/FewX)| [Paper](https://arxiv.org/abs/1908.01998). See [FewX](FewX) for details in implementations.


Follow the instructions in each folder for details in implementation/training/evaluations.

# Installation

Both faster-rcnn finetuning and few-shot approaches are built on top of [detectron2](https://github.com/facebookresearch/detectron2). Please follow the instruction in the offical page to[install detectron2] (https://detectron2.readthedocs.io/en/latest/tutorials/install.html).
You will also need to install appropriate pytorch and torchvision versions.
This repo is tested with `pytorch 1.10.1+cu111` and `detecton2 v0.6`

# Dataset and Annotations
Download the dataset and annotation from [Drive](https://drive.google.com/drive/folders/1r_cQSwtUteatAN8jsBpvhQ8013XW5l_e?usp=sharing) and extract into an appropriate folder.
You may need to point to this folder/update the dataset path in the dataloader/data preparation codes later. 
