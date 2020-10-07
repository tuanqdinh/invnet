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
from .iresnet_block import conv_iresnet_block
from .ops.mixup_ops import *

class conv_iResNet(nn.Module):
    def __init__(self, in_shape, nBlocks, nStrides, nChannels, init_ds=2, inj_pad=0,
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

        self.stack, self.in_shapes, self.final_shape = self._make_stack(nChannels, nBlocks, nStrides, in_shape, coeff, block,                                                                     actnorm, n_power_iter, nonlin)

        # make prior distribution
        self._make_prior(learn_prior)
        # make classifier
        self._make_classifier(self.final_shape, nClasses)
        assert (nClasses is not None or density_estimation), "Must be either classifier or density estimator"

    def _make_prior(self, learn_prior):
        dim = np.prod(self.in_shapes[0])
        self.prior_mu = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)
        self.prior_logstd = nn.Parameter(torch.zeros((dim,)).float(), requires_grad=learn_prior)

    def _make_classifier(self, final_shape, nClasses):
        if nClasses is None:
            self.logits = None
        else:
            self.bn1 = nn.BatchNorm2d(final_shape[0], momentum=0.9)
            self.logits = nn.Linear(final_shape[0], nClasses)
            self.bn2 = nn.BatchNorm2d(final_shape[0], momentum=0.9)
            self.logits2 = nn.Linear(final_shape[0], 2)

    def classifier(self, z):
        # class 1
        out = F.relu(self.bn1(z))
        out = F.avg_pool2d(out, out.size(2))
        out = out.view(out.size(0), out.size(1))
        # class 2
        out2 = F.relu(self.bn2(z))
        out2 = F.avg_pool2d(out2, out2.size(2))
        out2 = out2.view(out2.size(0), out2.size(1))

        return self.logits(out), self.logits2(out2)

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

    def get_in_shapes(self):
        return self.in_shapes

    def inspect_singular_values(self):
        i = 0
        j = 0
        params = [v for v in self.state_dict().keys()
                  if "bottleneck" in v and "weight_orig" in v
                  and not "weight_u" in v
                  and not "bn1" in v
                  and not "linear" in v]
        print(len(params))
        print(len(self.in_shapes))
        svs = []
        for param in params:
          input_shape = tuple(self.in_shapes[j])
          # get unscaled parameters from state dict
          convKernel_unscaled = self.state_dict()[param].cpu().numpy()
          # get scaling by spectral norm
          sigma = self.state_dict()[param[:-5] + '_sigma'].cpu().numpy()
          convKernel = convKernel_unscaled / sigma
          # compute singular values
          input_shape = input_shape[1:]
          fft_coeff = np.fft.fft2(convKernel, input_shape, axes=[2, 3])
          t_fft_coeff = np.transpose(fft_coeff)
          D = np.linalg.svd(t_fft_coeff, compute_uv=False, full_matrices=False)
          Dflat = np.sort(D.flatten())[::-1]
          print("Layer "+str(j)+" Singular Value "+str(Dflat[0]))
          svs.append(Dflat[0])
          if i == 2:
            i = 0
            j+= 1
          else:
            i+=1
        return svs

    def forward(self, x, targets=None, mixup=False, mixup_hidden=False, mixup_alpha=None, ignore_logdet=False):
        """ iresnet forward """
        if mixup_hidden:
            layer_mix = np.random.randint(0, len(self.stack) + 1)
        elif mixup:
            layer_mix = 0
        else:
            layer_mix = None

        if mixup_alpha is not None:
            lam = get_lambda(mixup_alpha)
            lam = torch.from_numpy(np.array([lam]).astype('float32')).cuda()
            lam = Variable(lam)

        if targets is not None :
            target_reweighted = to_one_hot(targets, self.nClasses)
        else:
            target_reweighted = None

        z = x
        if layer_mix == 0:
            z, target_reweighted = mixup_process(z, target_reweighted, lam=lam)

        if self.init_ds == 2:
            z = self.init_squeeze.forward(z)

        if self.ipad != 0:
            z = self.inj_pad.forward(z)

        for idx, block in enumerate(self.stack):
            # if debug:
                # print([torch.norm(p.detach()).cpu().numpy().item() for p in block.actnorm.parameters()] )
            z, _ = block(z, ignore_logdet=ignore_logdet)
            # self.print_stats(z, debug)
            if idx + 1 == layer_mix:
                z, target_reweighted = mixup_process(z, target_reweighted, lam=lam)

        logits = self.classifier(z)
        # self.print_stats(logits, debug)
        return logits, z, target_reweighted

    def inverse(self, z, max_iter=10):
        """ iresnet inverse """
        with torch.no_grad():
            x = z
            for i in range(len(self.stack)):
                x = self.stack[-1 - i].inverse(x, maxIter=max_iter)

            if self.ipad != 0:
                x = self.inj_pad.inverse(x)

            if self.init_ds == 2:
                x = self.init_squeeze.inverse(x)
        return x

    def sample(self, batch_size, max_iter=10):
        """sample from prior and invert"""
        with torch.no_grad():
            # only send batch_size to prior, prior has final_shape as attribute
            samples = self.prior().rsample((batch_size,))
            samples = samples.view((batch_size,) + self.final_shape)
            return self.inverse(samples, max_iter=max_iter)

    def set_num_terms(self, n_terms):
        for block in self.stack:
            block.numSeriesTerms = n_terms
