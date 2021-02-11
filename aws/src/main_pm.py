from __future__ import print_function

import sys, os
import math
import threading
import argparse
import time

import numpy as np
from mpi4py import MPI

import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from worker_pm import Worker
from master_pm import Master
from ops import load_data

import warnings
warnings.filterwarnings("ignore")

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    # Training settings
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--max-steps', type=int, default=10000, metavar='N',
                        help='the maximum number of iterations')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--network', type=str, default='LeNet', metavar='N',
                        help='which kind of network we are going to use, support LeNet and ResNet currently')
    parser.add_argument('--mode', type=str, default='normal', metavar='N',
                        help='determine if we kill the stragglers or just implement normal training')
    parser.add_argument('--kill-threshold', type=float, default=7.0, metavar='KT',
                        help='timeout threshold which triggers the killing process (default: 7s)')
    parser.add_argument('--comm-type', type=str, default='Bcast', metavar='N',
                        help='which kind of method we use during the mode fetching stage')
    parser.add_argument('--num-aggregate', type=int, default=5, metavar='N',
                        help='how many number of gradients we wish to gather at each iteration')
    parser.add_argument('--eval-freq', type=int, default=50, metavar='N',
                        help='it determines per how many step the model should be evaluated')
    parser.add_argument('--train-dir', type=str, default='output/models/', metavar='N',
                        help='directory to save the temp model during the training process for evaluation')
    parser.add_argument('--compress-grad', type=str, default='compress', metavar='N',
                        help='compress/none indicate if we compress the gradient matrix before communication')
    parser.add_argument('--enable-gpu', type=bool, default=False, help='whether to use gradient approx method')


    # addd fusion net opts
    parser.add_argument('--isTrain', action='store_false')
    parser.add_argument('--config_file', default='../config', type=str, help='dataset')
    parser.add_argument('--iname', default='mixup', type=str, help='dataset')
    parser.add_argument('--fname', default='pix2pix', type=str, help='dataset')

    parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
    parser.add_argument('--data_dir', default='../data/')
    parser.add_argument('--save_dir', default='../results',
                        type=str, help='directory to save results')

    ## training
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')

    # optimization
    parser.add_argument('--optimizer', default="sgd", type=str,
                        help="optimizer", choices=["adam", "adamax", "sgd"])
    parser.add_argument('--weight_decay', default=5e-4,
                        type=float, help='coefficient for weight decay')
    parser.add_argument('--drop_rate', default=0.1,
                        type=float, help='dropout rate')
    parser.add_argument('-drop_two', '--drop_two', dest='drop_two',
                        action='store_true', help='2d dropout on')
    parser.add_argument('-nesterov', '--nesterov', dest='nesterov', action='store_true',
                        help='nesterov momentum')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--niter', type=int, default=200, help='# of iter at starting learning rate')
    parser.add_argument('--niter_decay', type=int, default=200, help='# of iter to linearly decay learning rate to zero')
    parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy: lambda|step|plateau|cosine')
    parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
    parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count')

    # flags
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--flag_retrain', action='store_true', help='resume')
    parser.add_argument('--flag_plot', action='store_true', help='resume')
    parser.add_argument('--flag_test', action='store_true', help='test')
    parser.add_argument('--flag_val', action='store_true', help='val')
    parser.add_argument('--flag_overfit', action='store_true', help='Overfit')
    parser.add_argument('--resume', default=0, type=int, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--resume_g', default=0, type=int, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--flag_reg', action='store_true')
    parser.add_argument('--plot_fused', action='store_true')


    # nservers
    parser.add_argument('--nactors', type=int, default=2, help='n-servers')
    parser.add_argument('--reduction', type=str, default='mean', help='reduction')


    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')

    ###-------------- inet ----------------####
    # architectures
    # --nBlocks 7 7 7 --nStrides 1 2 2 --nChannels 32 64 128 --coeff 0.9 --batch 128 --init_ds 1 --inj_pad 0 --powerIterSpectralNorm 5 --nonlin elu --optimizer sgd --vis_server localhost  --vis_port 8097
    # data and paths
    parser.add_argument('--nClasses', nargs='+', type=int, default=10)
    parser.add_argument('--nBlocks', nargs='+', type=int, default=[7, 7, 7])
    parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 2, 2])
    parser.add_argument('--nChannels', nargs='+', type=int, default=[32, 64, 128])
    parser.add_argument('--inj_pad', default=0, type=int, help='initial inj padding')
    parser.add_argument('--coeff', default=0.9, type=float, help='contraction coefficient for linear layers')

    # networks
    parser.add_argument('-multiScale', '--multiScale', dest='multiScale', action='store_true', help='use multiscale')
    parser.add_argument('-fixedPrior', '--fixedPrior', dest='fixedPrior', action='store_true',
                        help='use fixed prior, default is learned prior')
    parser.add_argument('-noActnorm', '--noActnorm', dest='noActnorm', action='store_true',
                        help='disable actnorm, default uses actnorm')
    parser.add_argument('--nonlin', default="elu", type=str,
                        choices=["relu", "elu", "selu", "sorting", "softplus", "leaky"])

    #training
    parser.add_argument('--init_batch', default=1024,
                        type=int, help='init batch size')
    parser.add_argument('--init_ds', default=1, type=int,
                        help='initial downsampling')
    parser.add_argument('--warmup_epochs', default=10,
                        type=int, help='epochs for warmup')
    parser.add_argument('--eps', default=0.01, type=float)

    # modes
    parser.add_argument('--sample_fused', action='store_true')
    parser.add_argument('--eval_fusion', action='store_true')
    parser.add_argument('--eval_inv', action='store_true')
    parser.add_argument('--eval_distill', action='store_true')
    parser.add_argument('--test_latent', action='store_true')
    parser.add_argument('--plotTnse', action='store_true')

    parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation',
                        action='store_true', help='perform density estimation')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-interpolate', '--interpolate',
                        dest='interpolate', action='store_true', help='train iresnet')
    # parser.add_argument('-norm', '--norm', dest='norm', action='store_true',
    #                     help='compute norms of conv operators')
    parser.add_argument('--norm', default='batch', type=str)
    parser.add_argument('-analysisTraceEst', '--analysisTraceEst', dest='analysisTraceEst', action='store_true',
                        help='analysis of trace estimation')

    parser.add_argument('--numTraceSamples', default=1, type=int,
                        help='number of samples used for trace estimation')
    parser.add_argument('--numSeriesTerms', default=1, type=int,
                        help='number of terms used in power series for matrix log')
    parser.add_argument('--powerIterSpectralNorm', default=1, type=int,
                        help='number of power iterations used for spectral norm')

    ### mixup
    parser.add_argument('--mixup_alpha', default=1, type=float, help='dropout rate')
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup_hidden', action='store_true')
    parser.add_argument('--combined_loss', action='store_true')
    parser.add_argument('--concat_input', action='store_true', help='resume')

    # servers
    parser.add_argument('--vis_port', default=8097,
                        type=int, help="port for visdom")
    parser.add_argument('--vis_server', default="localhost",
                        type=str, help="server for visdom")

    # others
    parser.add_argument('--log_every', default=10,
                        type=int, help='logs every x iters')
    parser.add_argument('-log_verbose', '--log_verbose', dest='log_verbose', action='store_true',
                        help='verbose logging: sigmas, max gradient')
    parser.add_argument('-deterministic', '--deterministic', dest='deterministic', action='store_true',
                        help='fix random seeds and set cuda deterministic')
    parser.add_argument('--extension', default='.npy', type=str, help='extension')

    ###-------------- fnet ----------------####
    parser.add_argument('--lamb', type=float, default=100, help='weight on L1 term in objective')
    parser.add_argument('--lambda_distill', type=float, default=100, help='weight on L1 term in objective')
    parser.add_argument('--lambda_L1', type=float, default=100, help='weight on L1 term in objective')

    ### Pix2PixModel
    """Define the common options that are used in both training and test."""
    # basic parameters
    parser.add_argument('--gpu_ids', type=str, default='0,', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--checkpoints_dir', type=str, default='../results/checkpoints', help='models are saved here')
    # model parameters
    parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
    parser.add_argument('--netD', type=str, default='pixel', help='specify discriminator architecture [basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
    parser.add_argument('--netG', type=str, default='unet_128', help='specify generator architecture [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
    # parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
    parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')


    args = parser.parse_args()
    return args

if __name__ == "__main__":
    # this is only a simple test case
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    args = add_fit_args(argparse.ArgumentParser(description='PyTorch MNIST Single Machine Test'))

    if args.enable_gpu:
        args.gpu_ids = [0]
        args.device = torch.device("cuda")
    else:
        args.gpu_ids = []
        args.device = torch.device("cpu")

    args.nactors = world_size - 2
    args.batch_size = args.nactors

    args.noActnorm = True
    args.iname = "latency"
    args.inet_name = "iresnet-" + args.iname
    args.fnet_name = "fnet"
    checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
    inet_sample_dir = os.path.join(args.save_dir, 'samples/inet')
    args.inet_save_dir = os.path.join(checkpoint_dir, args.inet_name)
    args.fnet_save_dir = os.path.join(checkpoint_dir, args.fnet_name)

    kwargs_master = {'batch_size':args.batch_size, 
                'learning_rate':args.lr, 
                'max_epochs':args.epochs, 
                'momentum':args.momentum, 
                'network':args.network, 
                'comm_method':args.comm_type, 
                'kill_threshold': args.num_aggregate, 
                'timeout_threshold':args.kill_threshold, 
                'eval_freq':args.eval_freq, 
                'train_dir':args.train_dir, 
                'max_steps':args.max_steps, 
                'compress_grad':args.compress_grad,
                'device':args.device}

    kwargs_worker = {'batch_size':args.batch_size, 
                'learning_rate':args.lr, 
                'max_epochs':args.epochs, 
                'momentum':args.momentum, 
                'network':args.network,
                'comm_method':args.comm_type, 
                'kill_threshold':args.kill_threshold, 
                'eval_freq':args.eval_freq, 
                'train_dir':args.train_dir, 
                'max_steps':args.max_steps, 
                'compress_grad':args.compress_grad, 
                'device':args.device}


    # load data here 
    print('#-epochs: ', args.epochs)
    num_iters = args.epochs
    C, D, W = 3, 32, 32
    dataloader = load_data(args.batch_size)
    iters = iter(dataloader)
    args.nactors = world_size -2 
    M = args.batch_size // args.nactors
    N = M * args.nactors
    if rank == 0:
        master_node = Master(comm=comm, **kwargs_master)
        master_node.build_model(args)
        print("Master {}/{} starts".format(0, world_size))
        for i in range(num_iters):
            # add delay
            data = iters.next()[0]
            data = data[:N, ...].view(M, args.nactors, C, D, W)
            batch_data = data.numpy().astype(np.float32)
            # batch_da in cpu [M, A, C, W, D]
            master_node.run(batch_data)
            print('Done batch', i)

        print("Done sending messages to workers!")
        results, data = master_node.eval_latency()
        name = 'pm2_latency_{}_{}'.format(args.nactors, args.epochs)
        np.savetxt('{}.csv'.format(name), np.asarray(results), delimiter=',', fmt='%.5f')
        np.set_printoptions(formatter={'float': lambda x: "{:.5}".format(x)})
        names = ['latency', 'recon', 'inference', 'fusion']
        for i in range(len(results)):
            print(names[i], results[i])
        np.save(name, data)
    else:
        wk = Worker(comm=comm, **kwargs_worker)
        print("Worker: {}/{} starts".format(wk.rank, wk.world_size-1))
        if rank == 1:
            wk.build_model(args, fusion=True)
            for i in range(num_iters):
                wk.run(delay=-1)
        else:
            wk.build_model(args, fusion=False)
            for i in range(num_iters):
                off = i % args.nactors + 2 # worker to be delay
                wk.run(delay=off)