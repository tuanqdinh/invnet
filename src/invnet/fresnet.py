"""
    Code for "Invertible Residual Networks"
    @author: Tuan Dinh
    Edit from @
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from .ops.model_utils import squeeze as Squeeze
from .ops.model_utils import injective_pad
from .fresnet_block import conv_iresnet_block
from .ops.mixup_ops import *

class conv_iResNet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels, nactors, input_nc, init_ds=2, inj_pad=0,
                 coeff=.9, density_estimation=False, nClasses=None,
                 numTraceSamples=1, numSeriesTerms=1,
                 n_power_iter=5,
                 block=conv_iresnet_block,
                 actnorm=True, learn_prior=True,
                 nonlin="relu"):
        super(conv_iResNet, self).__init__()
        assert len(nBlocks) == len(nStrides) == len(nChannels)
        assert init_ds in (1, 2), "can only squeeze by 2"
        self.init_ds = init_ds
        self.ipad = inj_pad
        self.nBlocks = nBlocks
        self.density_estimation = density_estimation
        self.nClasses = nClasses
        # parameters for trace estimation
        self.numTraceSamples = numTraceSamples if density_estimation else 0
        self.numSeriesTerms = numSeriesTerms if density_estimation else 0
        self.n_power_iter = n_power_iter

        print('')
        print(' == Building iResNet %d == ' % (sum(nBlocks) * 3 + 1))
        self.init_squeeze = Squeeze(self.init_ds)
        self.inj_pad = injective_pad(inj_pad)
        if self.init_ds == 2:
           in_shape = downsample_shape(in_shape)
        in_shape = (in_shape[0] + inj_pad, in_shape[1], in_shape[2])  # adjust channels

        self.conv2d = nn.Conv2d(nactors * input_nc, in_shape[0], kernel_size=3, stride=1, padding=1)
        self.stack, self.in_shapes, self.final_shape = self._make_stack(nChannels, nBlocks, nStrides, in_shape, coeff, block,                                                                     actnorm, n_power_iter, nonlin)

    def _make_stack(self, nChannels, nBlocks, nStrides, in_shape, coeff, block,
                    actnorm, n_power_iter, nonlin):
        """ Create stack of iresnet blocks """
        block_list = nn.ModuleList()
        in_shapes = []
        for i, (int_dim, stride, blocks) in enumerate(zip(nChannels, nStrides, nBlocks)):
            for j in range(blocks):
                in_shapes.append(in_shape)
                block_list.append(block(in_shape, int_dim,
                                        numTraceSamples=self.numTraceSamples,
                                        numSeriesTerms=self.numSeriesTerms,
                                        stride=(stride if j == 0 else 1),  # use stride if first layer in block else 1
                                        input_nonlin=(i + j > 0),  # add nonlinearity to input for all but fist layer
                                        coeff=coeff,
                                        actnorm=actnorm,
                                        n_power_iter=n_power_iter,
                                        nonlin=nonlin))
                if stride == 2 and j == 0:
                    in_shape = downsample_shape(in_shape)

        return block_list, in_shapes, in_shape

    def forward(self, x, ignore_logdet=False):
        """ iresnet forward """
        if self.init_ds == 2:
            x = self.init_squeeze.forward(x)

        z = self.conv2d(x)
        for block in self.stack:
            z, _ = block(z, ignore_logdet=ignore_logdet)

        return z


    def get_in_shapes(self):
        return self.in_shapes
