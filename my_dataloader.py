#!/bin/python3

import torch
import torch.utils.data
import torchvision as tv
import cv2
import numpy
import os
import random
from pathlib import PurePosixPath
from pathlib import Path
import itertools

import yaml
import json
import sys

import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image
import multiprocessing as mp

from ultralytics.utils.ops import xywhn2xyxy

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, input_transform, output_transform):
        super().__init__()

        self.input_transform = input_transform
        self.output_transform = output_transform

    def __getitem__(self, item):
        return 0

    def __len__(self):
        return self.dataset_size

if __name__ == "__main__":
    root_path = Path("xxx")

    with open(root_path / "spec_labels.txt","rt") as f:
        spec_labels = [[float(x_item) for x_item in x.split()] for x in f.read().strip().splitlines() if len(x)]
    
    stop_flag = 0
    spec_labels = numpy.array(spec_labels)
    for spec_label in spec_labels:
        i_bottom = root_path / "images" / Path(str(int(spec_label[0]))+".jpg")
        i_top = root_path / "images" / Path(str(int(spec_label[1]))+".jpg")
        top_image = cv2.imread(str(i_top))
        bottom_image = cv2.imread(str(i_bottom))
        labels = [int(x) for x in xywhn2xyxy(spec_label[2:6], w=top_image.shape[1], h=top_image.shape[0])]
        #print(labels)

        new_size = (128,128)
        top_image_resized = cv2.resize(top_image[labels[1]:labels[3],labels[0]:labels[2]], new_size)
        bottom_image_resized = cv2.resize(bottom_image[labels[1]:labels[3],labels[0]:labels[2]], new_size)

        top_bottom = cv2.hconcat([top_image_resized,bottom_image_resized])

        cv2.imshow("top_bottom", top_bottom)

        key = cv2.waitKey(1-stop_flag)
        if key == 27:
            break
        elif key == 32:
            stop_flag = 1 - stop_flag

        #rect = xywhn2xyxy()
        #print(f'{i_top}, {i_bottom}')
