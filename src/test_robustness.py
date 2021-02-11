"""
	@author xxx xxx@cs.xxx.edu
	@date 02/14/2020
"""
from init_fusionnet import *
from fusionnet.pix2pix import Pix2PixModel
from ops import load_inet, evaluate_unet

import os
import sys
import torch as ch
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from robust_representations.robust.robustness import model_utils, datasets
import torchvision


# Constants
DATA = 'CIFAR' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
device = ch.device("cuda:0")
dataset_function = getattr(datasets, DATA)
dataset = dataset_function('../data/')
model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': f'./models/{DATA}.pt'
}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
model = model.to(device)
model.eval()
inet = model 

# Fusion Net
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids=[0]
args.pool_size=0
args.lambda_L1 = args.lamb

args.isTrain = False
fnet = Pix2PixModel(args)
init_epoch = int(args.resume_g)
fnet.load_networks(epoch=init_epoch)

if args.flag_test:
	print('Evaluate on Test Set')
	data_loader = test_loader
elif args.flag_val:
	print('Evaluate on Validation Set')
	data_loader = val_loader
else:
	print('Evaluate on Train Set')
	data_loader = train_loader

criterion = torch.nn.L1Loss()
lossf, corrects = evaluate_unet(data_loader, fnet, inet, criterion, device)
print('L1 {:.4f} Acc1: {:.4f}'.format(lossf, corrects))
