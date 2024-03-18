#!/bin/python3

import torch
import torch.utils.data
import cv2
import numpy
from pathlib import Path
import torchvision.transforms as T
from ultralytics.utils.ops import xywhn2xyxy

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, labels_filename, new_size, transform = None):
        super().__init__()

        with open(labels_filename) as f:
            self.spec_labels = numpy.array([[float(x_item) for x_item in x.split()] for x in f.read().strip().splitlines() if len(x)])

        self.root_dataset_path = Path(labels_filename).parent
        self.new_size = new_size
        self.transform = transform


    def __getitem__(self, item):
        spec_label = self.spec_labels[item]

        i_bottom = self.root_dataset_path / "images" / Path(str(int(spec_label[0]))+".jpg")
        i_top = self.root_dataset_path / "images" / Path(str(int(spec_label[1]))+".jpg")
        top_image = cv2.imread(str(i_top))
        bottom_image = cv2.imread(str(i_bottom))
        labels = [int(x) for x in xywhn2xyxy(spec_label[2:6], w=top_image.shape[1], h=top_image.shape[0])]
        top_image_resized = cv2.resize(top_image[labels[1]:labels[3],labels[0]:labels[2]], self.new_size)
        bottom_image_resized = cv2.resize(bottom_image[labels[1]:labels[3],labels[0]:labels[2]], self.new_size)
        image_cake_tensor = T.ToTensor()(numpy.concatenate([top_image_resized, bottom_image_resized],axis=2))
        return self.transform(image_cake_tensor) if self.transform else image_cake_tensor, spec_label[6]

    def __len__(self):
        return len(self.spec_labels)

if __name__ == "__main__":
    root_path = Path("xxx")

    transforms = T.Compose([
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomResizedCrop((128, 128), antialias=True),
        T.GaussianBlur(9)]
    )

    ds = MyDataset(root_path / "spec_labels.txt", (128,128), transform=transforms)
    val_loader = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=1, shuffle=True)
    stop_flag = 1

    with torch.no_grad():
        for batch, (X,y) in enumerate(val_loader):
            print(f'{X.shape} {y}')
            images = []
            for channel in X[0].numpy():
                to_show = (channel*255).astype(numpy.uint8)
                images.append(to_show)

            image_rows = []
            for i in range(0, len(images) // 3):
                image_rows.append(cv2.hconcat(images[3*i:3*i+3]))
            
            cv2.imshow('channels', cv2.vconcat(image_rows))
            key = cv2.waitKey(1-stop_flag)
            if key == 27:
                break
            elif key == 32:
                stop_flag = 1 - stop_flag

