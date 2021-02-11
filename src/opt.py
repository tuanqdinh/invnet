"""
	@author xxx xxx@cs.xxx.edu
	@date 08/14/2019
"""

import argparse

parser = argparse.ArgumentParser(description='Train ifnet')

## ------------ shared ------------###
## paths
parser.add_argument('--isTrain', action='store_false')
parser.add_argument('--config_file', default='../config', type=str, help='dataset')
parser.add_argument('--iname', default='mixup', type=str, help='dataset')
parser.add_argument('--fname', default='pix2pix', type=str, help='dataset')

parser.add_argument('--dataset', default='cifar10', type=str, help='dataset')
parser.add_argument('--data_dir', default='../data/')
parser.add_argument('--save_dir', default='../results',
                    type=str, help='directory to save results')

## training
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch_size', type=int, default=64, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', default=200, type=int, help='number of epochs')

parser.add_argument('--log_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=10)

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
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--bottleneck_mult', default=4, type=int, help='bottleneck multiplier')
###-------------- inet ----------------####
# architectures
# --nBlocks 7 7 7 --nStrides 1 2 2 --nChannels 32 64 128 --coeff 0.9 --batch 128 --init_ds 1 --inj_pad 0 --powerIterSpectralNorm 5 --nonlin elu --optimizer sgd --vis_server localhost  --vis_port 8097
# data and paths
parser.add_argument('--nClasses', nargs='+', type=int, default=10)
parser.add_argument('--nBlocks', nargs='+', type=int, default=[7, 7, 7])
parser.add_argument('--nStrides', nargs='+', type=int, default=[1, 2, 2])
parser.add_argument('--nChannels', nargs='+', type=int, default=[2, 8, 32])
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
# dataset parameters
parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
# additional parameters
parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

# visdom and HTML visualization parameters
parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
# network saving and loading parameters
parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
# training parameters
parser.add_argument('--n_epochs', type=int, default=50, help='number of epochs with the initial learning rate')
parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
parser.add_argument('--pool_size', type=int, default=0, help='the size of image buffer that stores previously generated images')

args = parser.parse_args()
# print(args)
