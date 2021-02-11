import torch, os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from codednet.invnet.iresnet import conv_iResNet as iResNet
from codednet.fusionnet.pix2pix import Pix2PixModel

from sys import getsizeof

class ModelBuffer(object):
    def __init__(self, num_nodes, in_shape):
        """
        this class is used to save model weights received from parameter server
        current step for each layer of model will also be updated here to make sure
        the model is always up-to-date
        """
        super(ModelBuffer, self).__init__()
        self.recv_buf = [bytearray(getsizeof(np.zeros(in_shape))) for _ in range(num_nodes)]
        self.num_nodes, self.in_shape = num_nodes, in_shape

    def reset(self):
        for i in range(self.num_nodes):
            self.recv_buf[i] = np.empty(self.in_shape, dtype=np.float32)

def load_data(batch_size):
    
    means = (0.4914, 0.4822, 0.4465)
    stds =  (0.2023, 0.1994, 0.2010)
    test_chain = [transforms.ToTensor()]
    clf_chain = [transforms.Normalize(means, stds)]
    transform_test = transforms.Compose(test_chain + clf_chain)
    testset = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transform_test)
    return DataLoader(testset, batch_size=batch_size, shuffle=True, drop_last=True)

def load_inet(args):
    in_shape = (3, 32, 32)
    inet = iResNet(nBlocks=args.nBlocks,
                    nStrides=args.nStrides,
                    nChannels=args.nChannels,
                    nClasses=args.nClasses,
                    init_ds=args.init_ds,
                    inj_pad=args.inj_pad,
                    in_shape=in_shape,
                    coeff=args.coeff,
                    numTraceSamples=args.numTraceSamples,
                    numSeriesTerms=args.numSeriesTerms,
                    n_power_iter = args.powerIterSpectralNorm,
                    density_estimation=args.densityEstimation,
                    actnorm=(not args.noActnorm),
                    learn_prior=(not args.fixedPrior),
                    nonlin=args.nonlin).to(args.device)
    inet.eval()
    return inet

def load_fnet(args):
    args.norm='batch'
    args.dataset_mode='aligned'
    args.pool_size=0
    args.lambda_L1 = args.lamb
    args.isTrain = False
    fnet = Pix2PixModel(args)
    fnet.netG.eval()
    return fnet
