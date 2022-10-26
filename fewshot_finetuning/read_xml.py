#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 18:56:06 2022

@author: dm0051
"""
import xml.etree.ElementTree as ET
import glob
import random 


annotations = glob.glob('/vol/research/deepdiscover/images/Paisley/*.xml')
annotations = [x.split('/')[-1].split('.')[0] for x in annotations]

n = len(annotations)

random.shuffle(annotations)

train = annotations[:n//2]
test = annotations[n//2:]

with open('Paisley_train.txt', 'w') as f:
    f.write('\n'.join(train))
   
with open('Paisley_test.txt', 'w') as f:
    f.write('\n'.join(test))
    
    



xml_file = annotations[1]

def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)

        list_with_single_boxes = [xmin, ymin, xmax, ymax]
        list_with_all_boxes.append(list_with_single_boxes)

    return filename, list_with_all_boxes

name, boxes = read_content("file.xml")