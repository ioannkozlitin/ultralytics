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
    
    spec_labels = numpy.array(spec_labels)
    for spec_label in spec_labels:
        i_bottom = root_path / "images" / Path(str(int(spec_label[0]))+".jpg")
        i_top = root_path / "images" / Path(str(int(spec_label[1]))+".jpg")
        top_image = cv2.imread(str(i_top))
        labels = xywhn2xyxy(spec_label[2:6], w=top_image.shape[1], h=top_image.shape[0])
        print(labels)

        cv2.imshow("image",top_image)
        if cv2.waitKey(1)==27:
            break

        #rect = xywhn2xyxy()
        #print(f'{i_top}, {i_bottom}')
