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
    def __init__(self, labels_filename, new_size):
        super().__init__()

        with open(labels_filename) as f:
            self.spec_labels = numpy.array([[float(x_item) for x_item in x.split()] for x in f.read().strip().splitlines() if len(x)])

        self.new_size = new_size


    def __getitem__(self, item):
        spec_label = self.spec_labels[item]

        i_bottom = root_path / "images" / Path(str(int(spec_label[0]))+".jpg")
        i_top = root_path / "images" / Path(str(int(spec_label[1]))+".jpg")
        top_image = cv2.imread(str(i_top))
        bottom_image = cv2.imread(str(i_bottom))
        labels = [int(x) for x in xywhn2xyxy(spec_label[2:6], w=top_image.shape[1], h=top_image.shape[0])]
        top_image_resized = cv2.resize(top_image[labels[1]:labels[3],labels[0]:labels[2]], self.new_size)
        bottom_image_resized = cv2.resize(bottom_image[labels[1]:labels[3],labels[0]:labels[2]], self.new_size)        
        return [top_image_resized, bottom_image_resized]

    def __len__(self):
        return len(self.spec_labels)

if __name__ == "__main__":
    root_path = Path("xxx")

    ds = MyDataset(root_path / "spec_labels.txt", (128,128))
    val_loader = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=1, shuffle=True)
    stop_flag = 1

    with torch.no_grad():
        for batch, item in enumerate(val_loader):
            print(item[0].numpy().shape)
            top_bottom = cv2.hconcat([item[0].numpy()[0], item[1].numpy()[0]])
            cv2.imshow("top_bottom", top_bottom)
            
            key = cv2.waitKey(1-stop_flag)
            if key == 27:
                break
            elif key == 32:
                stop_flag = 1 - stop_flag

