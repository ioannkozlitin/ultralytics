#!/bin/python3

import torch
import sys
import json
import os
from pathlib import Path

from my_dataloader import MyDataset
from nntrain import SmokeCnnModel

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as train_settings_file:
            nn_train_settings=json.load(train_settings_file)
    else:
        print("Use: " + sys.argv[0] + " train_local.json\n")
        exit(1)
        
    if len(sys.argv) > 2:
        versiontag = sys.argv[2]
    else:
    	versiontag = ""
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    model_dir = os.path.expanduser(nn_train_settings['model_dir'])
    model_filename = Path(model_dir) / (nn_train_settings['nn_model_name'] + versiontag + '.ckpt')

    model = SmokeCnnModel(frame_number=nn_train_settings['frame_number'])
    cp = torch.load(model_filename, map_location=device)
    val_loader = torch.utils.data.DataLoader(MyDataset(nn_train_settings['validation_labels_filename'], (128,128)),
                                                 num_workers=nn_train_settings['workers'],
                                                 batch_size=nn_train_settings['batch_size'], shuffle=False)

    model.load_state_dict(cp['state_dict'])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        s10 = 0
        s01 = 0
        s00 = 0
        s11 = 0
        s0 = 0
        s1 = 0
        y0 = 0
        y1 = 0
        for batch, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            pred = model.net(X)
            #for item in pred:
            #	if item[0]>item[1]:
            #		print(item)
            
            prob = torch.argmax(pred, dim=1)
            s10 = s10 + sum((y == 1) & (prob == 0))
            s01 = s01 + sum((y == 0) & (prob == 1))
            s00 = s00 + sum((y == 0) & (prob == 0))
            s11 = s11 + sum((y == 1) & (prob == 1))
            s1 = s1 + sum(prob == 1)
            s0 = s0 + sum(prob == 0)
            y1 = y1 + sum(y == 1)
            y0 = y0 + sum(y == 0)
            print(f'[{batch}]')
    f0 = s00 / y0
    f1 = s11 / y1
    print(f'\ns10 = {s10}\ns01 = {s01}\ns00 = {s00}\ns11 = {s11}')
    print(f'y0 = {y0}\ny1 = {y1}\ns0 = {s0}\ns1 = {s1}\n')
    print(f'f0 = {f0}\nf1 = {f1}\n')
    
