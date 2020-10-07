import torch, os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from invnet.iresnet import conv_iResNet as iResNet
from utils.provider import Provider
from utils.helper import Helper
from torch.utils.data import DataLoader
# from utils.tester import atanh

bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)
criterionKLD = nn.KLDivLoss()

def load_networks(net, resume, net_save_dir):
	if resume == 0:
		save_filename = 'current_net.pth'
	elif resume == 1:
		save_filename = 'best_net.pth'
	else:
		save_filename = '%s_net.pth' % (resume)
	net_path = os.path.join(net_save_dir, save_filename)

	init_epoch = 0
	if os.path.isfile(net_path):
		print("-- Loading checkpoint '{}'".format(net_path))

		state = torch.load(net_path)
		# net.load_state_dict(state)
		net.load_state_dict(state['model'])
		# init_epoch = state['epoch']
		return True, net
	else:
		print("--- No checkpoint found at '{}'".format(net_path))
		return False, None

def load_inet(args, device):
	# iResNet data
	trainset, testset, in_shape = Provider.load_data(args.dataset, args.data_dir)
	trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
	# testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
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
					nonlin=args.nonlin).to(device)
	init_batch = Helper.get_init_batch(trainloader, args.init_batch)
	print("initializing actnorm parameters...")
	with torch.no_grad():
		inet(init_batch.to(device), ignore_logdet=True)
	print("initialized")
	inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))

	if args.resume > 0:
		return load_networks(inet, args.resume, args.inet_save_dir)

	return True, inet


def evaluate_unet(data_loader, fnet, inet, criterion, device):
	lossf = []
	corrects = [0, 0]
	total = 0
	for _, (images, targets) in enumerate(data_loader):
		inputs = images.to(device)
		targets = targets.to(device)
		N, A, C, H, W = inputs.shape
		# inputs = inputs[:, torch.randperm(A), ...]
		fake = fnet.netG(inputs.view(N, C*A, H, W))
		loss_l1 = criterion(fake, targets)
		lossf.append(loss_l1.data.cpu().numpy())

		# Classification evaluation
		_, z_c, _ = inet(inputs.view(N * A, C, H, W))
		z_c = z_c.view(N, A, z_c.shape[1], z_c.shape[2], z_c.shape[3])
		z_0 = z_c[:, 0, ...]
		z_c = z_c[:, 1:, ...].sum(dim=1)
		# img_fused = atanh(fake)
		img_fused = fake # no tanh
		_, z_fused, _ = inet(img_fused)
		z_hat = A * z_fused - z_c

		out_hat = inet.module.classifier(z_hat)
		out = inet.module.classifier(z_0)
		for k in range(2):
			_, y = torch.max(out[k].data, 1)
			_, y_hat = torch.max(out_hat[k].data, 1)
			correct = y_hat.eq(y.data).sum().cpu().numpy()
			corrects[k] += correct
		total += N

	return np.mean(lossf), np.asarray(corrects) / total


def evaluate_dnet(data_loader, fnet, inet, device, nactors, nClasses):
	corrects = []
	A = nactors
	fnet.eval()
	for _, (inputs, labels) in enumerate(data_loader):
		inputs = inputs.to(device)
		labels = labels.to(device)
		N, C, H, W = inputs.shape
		M = N // A
		N = M * A
		inputs = inputs[:N, ...]
		labels = labels[:N, ...]

		logits, _, _ = inet(inputs)
		logits = logits.view(M, A, nClasses)
		logit_missed = logits[:, 0, :]
		logit_rest = logits[:, 1:, :].sum(dim=1)
		# fusion inputs
		x_g = inputs.view(M, A, C, H, W).view(M, A*C, H, W)
		logits_mean_g, _, _ = fnet(x_g)
		logits_missed_g = logits_mean_g * A - logit_rest

		_, y = torch.max(logit_missed.data, 1)
		_, y_hat = torch.max(logits_missed_g.data, 1)
		# dangerous => 0
		correct = y_hat.eq(y.data).sum().cpu().numpy() / M
		corrects.append(correct)

	return np.mean(corrects)

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
	"""
	Compute the knowledge-distillation (KD) loss given outputs, labels.
	"Hyperparameters": temperature and alpha
	NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
	and student expects the input tensor to be log probabilities! See Issue #2
	"""
	T = temperature # small T for small network student
	beta = (1. - alpha) * T * T # alpha for student: small alpha is better
	teacher_loss = criterionKLD(F.log_softmax(outputs/T, dim=1),
							 F.softmax(teacher_outputs/T, dim=1))
	# student_loss = F.cross_entropy(outputs, labels)
	student_loss = bce_loss(softmax(outputs), labels)
	KD_loss =  beta * teacher_loss + alpha * student_loss

	return KD_loss
