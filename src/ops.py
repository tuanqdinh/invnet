import torch, os
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from codednet.invnet.iRevNet import iRevNet as iResNet
from utils.provider import Provider
from utils.helper import Helper
from torch.utils.data import DataLoader
# from utils.tester import atanh

bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)
criterionKLD = nn.KLDivLoss()

def load_networks(net, resume, net_save_dir):
	save_filename = '%s_net.pth' % (resume)
	net_path = os.path.join(net_save_dir, save_filename)
	init_epoch = 0
	if os.path.isfile(net_path):
		print("-- Loading checkpoint '{}'".format(net_path))
		state = torch.load(net_path)
		net.load_state_dict(state['model'])
		return True, net
	else:
		print("--- No checkpoint found at '{}'".format(net_path))
		return False, None

def load_inet(args, device):
	in_shape=(1, 32, 32)
	args.nChannels = [2, 8, 32]
	args.bottleneck_mult = 1
	args.init_ds = 2
	inet = iResNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
					nChannels=args.nChannels, nClasses=10,
					init_ds=args.init_ds, dropout_rate=0.1, affineBN=True,
					in_shape=in_shape, mult=args.bottleneck_mult).to(device)
	inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))

	if args.resume > 0:
		return load_networks(inet, args.resume, args.inet_save_dir)

	return True, inet


def evaluate_unet(data_loader, fnet, inet, criterion, device):
	lossf = []
	corrects = 0
	total = 0
	inet.eval()
	fnet.netG.eval()
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
		_, y = torch.max(out.data, 1)
		_, y_hat = torch.max(out_hat.data, 1)
		correct = y_hat.eq(y.data).sum().cpu().numpy()
		corrects += correct / N
		total += 1

	return np.mean(lossf), corrects/total

def evaluate_unet_robustness(data_loader, fnet, inet, criterion, device):
	lossf = []
	corrects = 0
	total = 0
	inet.eval()
	fnet.netG.eval()
	for _, (images, targets) in enumerate(data_loader):
		inputs = images.to(device)
		targets = targets.to(device)
		N, A, C, H, W = inputs.shape
		# inputs = inputs[:, torch.randperm(A), ...]
		fake = fnet.netG(inputs.view(N, C*A, H, W))
		loss_l1 = criterion(fake, targets)
		lossf.append(loss_l1.data.cpu().numpy())

		# Classification evaluation
		with torch.no_grad():
			# z_c: 2048
			(_, z_c), _ = inet(inputs.view(N * A, C, H, W), with_latent=True)

		z_c = z_c.view(N, A, 2048)
		z_0 = z_c[:, 0, ...]
		z_c = z_c[:, 1:, ...].sum(dim=1)
		# img_fused = atanh(fake)
		img_fused = fake # no tanh
		(_, z_fused), _ = inet(img_fused, with_latent=True)
		z_hat = A * z_fused - z_c

		out_hat = inet.classify(z_hat)

		out = inet.classify(z_0)

		_, y = torch.max(out.data, 1)
		_, y_hat = torch.max(out_hat.data, 1)
		correct = y_hat.eq(y.data).sum().cpu().numpy()
		corrects += correct / N
		total += 1

	return np.mean(lossf), corrects/total


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
