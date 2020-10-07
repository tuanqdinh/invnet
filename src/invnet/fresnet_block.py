
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ops.spectral_norm_conv_inplace import spectral_norm_conv
from .ops.spectral_norm_fc import spectral_norm_fc
from .ops.model_utils import injective_pad, ActNorm2D, Split, MaxMinGroup
from .ops.model_utils import squeeze as Squeeze
from .ops.matrix_utils import exact_matrix_logarithm_trace, power_series_matrix_logarithm_trace

class conv_iresnet_block(nn.Module):
    def __init__(self, in_shape, int_ch, numTraceSamples=0, numSeriesTerms=0,
                 stride=1, coeff=.97, input_nonlin=True,
                 actnorm=True, n_power_iter=5, nonlin="elu"):
        """
        buid invertible bottleneck block
        :param in_shape: shape of the input (channels, height, width)
        :param int_ch: dimension of intermediate layers
        :param stride: 1 if no downsample 2 if downsample
        :param coeff: desired lipschitz constant
        :param input_nonlin: if true applies a nonlinearity on the input
        :param actnorm: if true uses actnorm like GLOW
        :param n_power_iter: number of iterations for spectral normalization
        :param nonlin: the nonlinearity to use
        """
        super(conv_iresnet_block, self).__init__()
        assert stride in (1, 2)
        self.stride = stride
        self.squeeze = Squeeze(stride)
        self.coeff = coeff
        self.numTraceSamples = numTraceSamples
        self.numSeriesTerms = numSeriesTerms
        self.n_power_iter = n_power_iter
        nonlin = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "softplus": nn.Softplus,
            "sorting": lambda: MaxMinGroup(group_size=2, axis=1)
        }[nonlin]

        # set shapes for spectral norm conv
        in_ch, h, w = in_shape

        layers = []
        if input_nonlin:
            layers.append(nonlin())

        in_ch = in_ch * stride**2
        kernel_size1 = 3 # kernel size for first conv
        layers.append(nn.Conv2d(in_ch, int_ch, kernel_size=kernel_size1, stride=1, padding=1))
        layers.append(nonlin())
        kernel_size2 = 1 # kernel size for second conv
        layers.append(nn.Conv2d(int_ch, int_ch, kernel_size=kernel_size2, padding=0))
        layers.append(nonlin())
        kernel_size3 = 3 # kernel size for third conv
        layers.append(nn.Conv2d(int_ch, in_ch, kernel_size=kernel_size3, padding=1))
        self.bottleneck_block = nn.Sequential(*layers)
        if actnorm:
            self.actnorm = nn.BatchNorm2d(in_ch)
        else:
            self.actnorm = None

    def forward(self, x, ignore_logdet=False):
        """ bijective or injective block forward """
        if self.stride == 2:
            x = self.squeeze.forward(x)

        x = self.actnorm(x)

        Fx = self.bottleneck_block(x)
        # Compute approximate trace for use in training
        if (self.numTraceSamples == 0 and self.numSeriesTerms == 0) or ignore_logdet:
            trace = torch.tensor(0.)
        else:
            trace = power_series_matrix_logarithm_trace(Fx, x, self.numSeriesTerms, self.numTraceSamples)

        # add residual to output
        y = Fx + x
        return y, trace
