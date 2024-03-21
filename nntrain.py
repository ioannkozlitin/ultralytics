#!/bin/python3

import torch
import torchmetrics
import torchvision as tv
from pathlib import Path
from pathlib import PurePosixPath
import pytorch_lightning as pl
import onnx
#from onnx_tf.backend import prepare
import tensorflow as tf
import shutil
import sys
import json
import os

from my_dataloader import MyDataset
from functools import partial

import torchvision.transforms as T

class SmokeCnnModel(torch.nn.Module):
    def __init__(self, frame_number):
        super().__init__()
        self.net = tv.models.mobilenet_v3_small(pretrained=False, num_classes=2)
        print(f'before: {self.net.features._modules["0"]}')

        self.net.features._modules['0'] = tv.ops.misc.Conv2dNormActivation(3*frame_number, 16,
                            kernel_size=3,
                            stride=2,
                            norm_layer=partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.01),
                            activation_layer=torch.nn.Hardswish)

        print(f'after: {self.net.features._modules["0"]}')


class SmokeCnn(pl.LightningModule):
    def __init__(self, batch_size, workers, balance,
                 frame_number, train_labels_filename, validation_labels_filename, new_size, transform = None):
        super().__init__()
        self.net = SmokeCnnModel(frame_number).net
        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.train_labels_filename = train_labels_filename
        self.validation_labels_filename = validation_labels_filename
        self.new_size = new_size
        self.transform = transform
        self.batch_size = batch_size
        self.workers = workers
        self.balance = balance
        self.w = 1
        self.w_norm = 1
        self.frame_number = frame_number

    def forward(self, x):
        y = self.net(x)
        return y

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(MyDataset(self.train_labels_filename, self.new_size, transform=self.transform),
                                                   num_workers=self.workers,
                                                   batch_size=self.batch_size, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(MyDataset(self.validation_labels_filename, self.new_size),
                                                 num_workers=self.workers,
                                                 batch_size=self.batch_size, shuffle=False)
        return val_loader

    def loss_fun(self, preds, y, prob):
        #loss = tv.ops.sigmoid_focal_loss(prob[:, 0], y.float(), 0.25, 4, "mean")
        #loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([3, 1], dtype=torch.float, device=self.device))(preds, y)
        y1 = 2*y-1
        yw = y*(1-self.w) + self.w
        nnorm = (0.5*(preds[:, 1]**2 + preds[:, 0]**2))**0.5
        loss = torch.sum(yw * ((preds[:, 1] / nnorm - y1)**2 + (preds[:, 0] / nnorm + y1)**2)) / self.w_norm
        return loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.net(x)

        prob = torch.nn.Softmax()(preds)
        acc = self.train_acc(prob, y)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        loss = self.loss_fun(preds, y, prob)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.net(x)

        prob = torch.nn.Softmax()(preds)
        acc = self.val_acc(prob, y)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        val_loss = self.loss_fun(preds, y, prob)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer

    def calculate_w(self):
        with torch.no_grad():
            y0 = 0
            y1 = 0
            for batch, (X, y) in enumerate(self.train_dataloader()):
                y1 = y1 + torch.sum(y == 1)
                y0 = y0 + torch.sum(y == 0)
            self.w = y1 / y0 * self.balance
            self.w_norm = self.batch_size * y1 / (y1 + y0) * (1 + self.balance)

            print(self.w, self.w_norm)


def train_core(train_settings):
    print(pl.__version__)
    nn_train_settings = train_settings
    print(nn_train_settings)

    transforms = T.Compose([
        T.RandomRotation(degrees=(-15, 15)),
        T.RandomResizedCrop((128, 128), antialias=True),
        T.GaussianBlur(9)]
    )

    module = SmokeCnn(train_labels_filename=nn_train_settings['train_labels_filename'],
                      validation_labels_filename=nn_train_settings['validation_labels_filename'],
                      batch_size=nn_train_settings['batch_size'],
                      workers=nn_train_settings['workers'],
                      balance=nn_train_settings['balance'],
                      frame_number=nn_train_settings['frame_number'],
                      new_size=(128,128),
                      transform=transforms)

    model_dir = os.path.expanduser(train_settings['model_dir'])
    model_filename = nn_train_settings['nn_model_name']
    early_stopping_patience = nn_train_settings['early_stopping_patience']


    model_checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_loss', mode='min', save_top_k=1, verbose=True,
                                         dirpath=model_dir, filename=model_filename)


    trainer = pl.Trainer(
        accelerator="auto",
        logger=pl.loggers.TensorBoardLogger('logs/', name=None, version="Smoke"),
        #resume_from_checkpoint=model_full_filename,
        callbacks=[
            pl.callbacks.early_stopping.EarlyStopping(monitor='val_loss', mode='min', patience=early_stopping_patience, verbose=True),
            model_checkpoint
        ]
    )



    if Path(trainer.logger.log_dir).exists():
        shutil.rmtree(trainer.logger.log_dir)

    module.calculate_w()
    trainer.fit(module)

    print(f'Best model {model_checkpoint.best_model_path}')

    #input_size = list(next(iter(module.train_dataloader()))[0].size())
    #input_size[0] = 1
    #dummy_input = torch.ones(input_size)

    #device = 'cpu'
    #best_model = SmokeCnnModel(nn_train_settings['frame_number'])
    #cp = torch.load(model_checkpoint.best_model_path, map_location=device)
    #best_model.load_state_dict(cp['state_dict'])
    #best_model = best_model.to(device)
    #best_model.eval()

    #input_names = ["input1"]
    #output_names = ["output1"]
    #dynamic_axes = {'input1': {0: 'batch_size'},  # variable length axes
    #                'output1': {0: 'batch_size'}}

    #torch.onnx.export(best_model.net, dummy_input, onnx_model_filename, verbose=True,
    #                  input_names=input_names,
    #                  output_names=output_names,
    #                  dynamic_axes=dynamic_axes)

    #onnx_model = onnx.load(onnx_model_filename)  # load onnx model
    #tf_rep = prepare(onnx_model)  # prepare tf representation
    #tf_rep.export_graph(tf_model_filename)  # export the model

    #tfmodel = tf.saved_model.load(tf_model_filename)
    #inference_fun = tfmodel.signatures["serving_default"]

    #np_dummy_input = dummy_input.numpy()
    #tf_dummy_input = tf.convert_to_tensor(np_dummy_input)

    #print('pytorch', best_model.net(dummy_input))
    #print('tensorflow ', inference_fun(tf_dummy_input))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as train_settings_file:
            train_core(json.load(train_settings_file))
    else:
        print("Use: " + sys.argv[0] + " train_local.json\n")
        exit(1)
