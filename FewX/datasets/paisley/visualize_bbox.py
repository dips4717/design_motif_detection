#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:34:13 2022

@author: dipu
"""

import cv2
import numpy as np
from os.path import join, isdir
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

def vis_image(im, bboxs, im_name):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    dpi = 300
    fig, ax = plt.subplots() 
    ax.imshow(im, aspect='equal') 
    plt.axis('off') 
    height, width, channels = im.shape 
    fig.set_size_inches(width/100.0/3.0, height/100.0/3.0) 
    plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
    plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
    plt.margins(0,0)
    # Show box (off by default, box_alpha=0.0)
    #for bbox in bboxs:
    bbox = [int(x) for x in bboxs]

    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1],
                      fill=False, edgecolor='r',
                      linewidth=0.5, alpha=1))
   
    output_name = os.path.basename(im_name)
    #plt.show()
    plt.savefig(im_name, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close('all')

support_df = pd.read_pickle("10_shot_support_df.pkl")

for ii, df in support_df.iterrows():
    im = cv2.imread(df.file_path)
    bboxs = df.support_box
    imname = df.file_path.split('/')[-2:]
    im_name = imname[0] + '_' + imname[1]
    vis_image(im, bboxs, im_name)
    