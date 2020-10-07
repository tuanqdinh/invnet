"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler

import numpy as np
import visdom, os, sys, time, pdb, random, json

from utils.helper import Helper, AverageMeter
from utils.plotter import Plotter
from utils.tester import iTester
from utils.provider import Provider
from opt import args
from invnet.iresnet import conv_iResNet as iResNet

import warnings
warnings.filterwarnings("ignore")
########## Setting ####################
device = torch.device("cuda:0")
use_cuda = torch.cuda.is_available()
cudnn.benchmark = True
TRAIN_FRACTION = 0.8

########## Paths ####################
with open(args.config_file, 'r') as infile:
	cfg = json.load(infile)

args.dataset = cfg["dataset"]
args.init_ds = cfg["init"]
args.nBlocks = cfg["nBlocks"]
args.noActnorm = cfg["noActnorm"]
args.nonlin = cfg["nonlin"]
args.inj_pad = cfg["inj_pad"]

if args.mixup:
	mixname = "mixup"
elif args.mixup_hidden:
	mixname = "mixhidden"
else:
	mixname = "vanilla"
if args.noActnorm:
	actname = "noact"
else:
	actname = "act"

args.iname = "{}-{}-{}-{}".format(args.dataset, args.nBlocks[0], mixname, actname)
args.inet_name = "inet-" + args.iname
args.fnet_name = "fnet-{}-{}-{}".format(args.fname, args.iname, args.nactors)


checkpoint_dir = os.path.join(args.save_dir, 'checkpoints')
inet_sample_dir = os.path.join(args.save_dir, 'samples/inet')
args.inet_save_dir = os.path.join(checkpoint_dir, args.inet_name)
args.fnet_save_dir = os.path.join(checkpoint_dir, args.fnet_name)
Helper.try_make_dir(args.save_dir)
Helper.try_make_dir(checkpoint_dir)
Helper.try_make_dir(inet_sample_dir)
Helper.try_make_dir(args.fnet_save_dir)
Helper.try_make_dir(args.inet_save_dir)

########### DATA #################
traindata, testset, in_shape = Provider.load_data(args.dataset, args.data_dir)
nsamples = len(traindata)
train_size = int(nsamples * TRAIN_FRACTION)
val_size = nsamples - train_size
trainset, valset = torch.utils.data.random_split(traindata, [train_size, val_size])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.dataset == 'cifar10':
	args.input_nc = 3
	args.output_nc = 3
	in_shape = (3, 32, 32)
else:
	args.input_nc = 1
	args.output_nc = 1
	in_shape = (1, 32, 32)


########## Loss ####################
bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)
criterionCE = nn.CrossEntropyLoss()
criterionL1 = nn.L1Loss()
criterionKLD = nn.KLDivLoss()


########## # Evaluation ####################
def analyse(args, model, in_shapes, trainloader, testloader, fuse_type=None):

	tester = iTester(model, args.inet_name, in_shapes, testloader, args.nactors, fuse_type=fuse_type)

	if args.evaluate:
		scores = tester.evaluate()
		print('Evaluate')
		print(scores)
		return True

	if args.eval_inv:
		tester.eval_inv()
		return True

	if args.plot_fused:
		sample_path = os.path.join(args.save_dir, 'samples')
		tester.plot_fused(sample_path)
		return True

	if args.sample_fused:
		fusion_data_dir = os.path.join(args.data_dir, "fusion/{}-{}".format(args.iname, args.nactors))
		Helper.try_make_dir(fusion_data_dir)

		train_path = os.path.join(fusion_data_dir, 'train.npy')
		test_path = os.path.join(fusion_data_dir, 'test.npy')
		val_path = os.path.join(fusion_data_dir, 'val.npy')

		print("Generating train set")
		tester.sample_fused(train_path, trainloader, niters=4)
		print("Generating test set")
		tester.sample_fused(test_path, testloader)
		print("Generating validation set")
		tester.sample_fused(val_path, valloader)
		return True

	if args.eval_fusion:
		from fusionnet.pix2pix import Pix2PixModel as FModel

		# if args.fname == 'pix2pix':
			# from fusionnet.pix2pix import Pix2PixModel as FModel
		# else:
			# from fusionnet.unet import UnetModel as FModel

		args.gpu_ids = [0]
		args.isTrain = False
		fmodel = FModel(args)
		fmodel.setup(args)
		fmodel.load_networks(epoch=int(args.resume_g))

		fnet = fmodel.netG
		fnet.eval()
		tester.evaluate_fgan(fnet)

		# sample_path = os.path.join(args.save_dir, 'samples')
		# tester.plot_reconstruction(fnet, args.fnet_name, sample_path)

		return True

	if args.eval_distill:
		from fusionnet.networks import define_G
		args.gpu_ids = [0]

		fnet = define_G(args.input_nc, args.output_nc, args.ngf, args.netG, args.norm, not args.no_dropout, args.init_type, args.init_gain, args.gpu_ids, nactors=args.nactors)

		save_filename = '%s_net.pth' % (args.resume_g)
		fnet_path = os.path.join(args.fnet_save_dir, save_filename)
		fnet.load_state_dict(torch.load(fnet_path))
		print("-- Loading checkpoint '{}'".format(fnet_path))

		fnet.eval()
		tester.evaluate_fgan(fnet)
		return True

	if args.test_latent:
		# from torch.utils.data.sampler import SubsetRandomSampler
		labels = testset.targets
		indices = np.argwhere(np.asarray(labels) == 1)
		indices = indices.reshape(len(indices))

		set = torch.utils.data.Subset(testset, indices)
		dataloader = torch.utils.data.DataLoader(set, batch_size=args.batch_size, shuffle=False, num_workers=2)

		sample_path = os.path.join(args.save_dir, 'samples')
		tester.plot_fused(sample_path, dataloader)
		return True

	if args.plotTnse:
		num_classes = 7
		labels = trainset.targets
		indices = np.argwhere(np.asarray(labels) < num_classes)
		indices = indices.reshape(len(indices))
		trainset = torch.utils.data.Subset(trainset, indices)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)
		Tester.plot_latent(model, trainloader, num_classes)
		return True

	if args.analysisTraceEst:
		Tester.anaylse_trace_estimation(model, testset, use_cuda, args.extension)
		return True

	if args.norm == 'True':
		svs = Tester.test_spec_norm(model, in_shapes, args.extension)
		svs = svs[0]
		print(np.min(svs), np.median(svs), np.max(svs))
		return True

	if args.interpolate:
		Tester.interpolate(model, testloader, testset, start_epoch, use_cuda, best_objective, args.dataset)
		return True

	return False
