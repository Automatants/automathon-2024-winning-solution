import pytorch_lightning as pl
import torch.nn as nn
import torch
import argparse
import yaml
import munch
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torchinfo
import os
import json

from torchvision.models import efficientnet_b0



if __name__ == '__main__':

    model = efficientnet_b0(pretrained=True) 
    x = torch.randn(64, 3, 16, 128, 128)
    print(x.shape)
    x2 = x[:,:,0,:,:]
    print(x2.shape)
    x3 = model(x2)
    print(x3.shape)
