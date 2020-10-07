"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

from init_invnet import *
from invnet.ops.mixup_ops import to_one_hot
from fusionnet.fnet import FnetModel
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../results/runs/' + args.fnet_name)

########################## Helpers ##################

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
	"""
	Compute the knowledge-distillation (KD) loss given outputs, labels.
	"Hyperparameters": temperature and alpha
	NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
	and student expects the input tensor to be log probabilities! See Issue #2
	"""
	T = temperature # small T for small network student
	beta = (1. - alpha) * T * T # alpha for student: small alpha is better
	teacher_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
							 F.softmax(teacher_outputs/T, dim=1))
	# student_loss = F.cross_entropy(outputs, labels)
	student_loss = bce_loss(softmax(outputs), labels)
	KD_loss =  beta * teacher_loss + alpha * student_loss

	return KD_loss

def atanh(x, eps=1e-4):
	return 0.5*torch.log((1+x + eps)/(1-x + eps))

########################## Load ##################

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
inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))
if args.resume > 0:
	save_filename = '%s_net.pth' % (args.resume)
	inet_path = os.path.join(args.inet_save_dir, save_filename)
	if os.path.isfile(inet_path):
		print("-- Loading checkpoint '{}'".format(inet_path))
		inet.load_state_dict(torch.load(inet_path))
	else:
		print("--- No checkpoint found at '{}'".format(inet_path))
		sys.exit('Done')


####################### Fusion ###########################
#defaults
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
args.pool_size=0
A = args.nactors

fnet = FnetModel(args)
fnet.setup(args)
init_epoch = int(args.resume_g)
if init_epoch > 0:
	print("-- Loading checkpoint '{}'")
	fnet.load_networks(epoch=init_epoch)

########################## Training ###################################
print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
total_steps = len(iter(trainloader))
for epoch in range(init_epoch + 1, init_epoch + 1 + args.epochs):
	inet.eval()
	start_time = time.time()
	lossf = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	for batch_idx, (inputs, labels) in enumerate(trainloader):
		inputs = Variable(inputs).cuda()
		labels = Variable(labels).cuda()
		N, C, H, W = inputs.shape
		M = N // A
		N = M * A

		_, z, _ = inet(inputs)
		N, Cz, Hz, Wz = z.shape
		z = z.view(M, A, Cz, Hz, Wz).mean(dim=1)
		x_targets = inet.module.inverse(z)
		out = inet.module.classifier(z)

		# fusion inputs
		x_g = inputs[:N, ...].view(M, A, C, H, W).view(M, A*C, H, W)

		# forward
		fnet.set_input(x_g, torch.tanh(x_targets))         # unpack data from dataset and apply preprocessing
		fnet.forward()
		# fnet.optimize_parameters()   # calculate loss functions, get gradients, update network weights
		# if batch_idx % 3 == 0:
		fnet.set_requires_grad(fnet.netD, True)  # enable backprop for D
		fnet.optimizer_D.zero_grad()     # set D's gradients to zero
		fnet.backward_D()                # calculate gradients for D
		fnet.optimizer_D.step()          # update D's weights
		# update G
		fnet.set_requires_grad(fnet.netD, False)  # D requires no gradients when optimizing G
		fnet.optimizer_G.zero_grad()        # set G's gradients to zero
		x_fused_g = atanh(fnet.fake_B)
		out_hat, z_hat, _ = inet(x_fused_g)
		# inet loss
		labels_reweighted = to_one_hot(labels[:N, ...], args.nClasses).view(M, A, args.nClasses).mean(dim=1)
		fnet.loss_G_distill =  fnet.opt.lambda_distill  * loss_fn_kd(out_hat, labels_reweighted, out, alpha=0.1, temperature=6)
		# First, G(A) should fake the discriminator
		fake_AB = torch.cat((fnet.real_A, fnet.fake_B), 1)
		pred_fake = fnet.netD(fake_AB)
		fnet.loss_G_GAN = fnet.criterionGAN(pred_fake, True)
		# Second, G(A) = B
		fnet.loss_G_L1 = fnet.criterionL1(fnet.fake_B, fnet.real_B) * fnet.opt.lambda_L1
		# combine loss and calculate gradients
		fnet.loss_G = fnet.loss_G_GAN + fnet.loss_G_L1 + fnet.loss_G_distill
		fnet.loss_G.backward()
		fnet.optimizer_G.step()

		# measure accuracy and record loss
		_, labels = torch.max(labels_reweighted.data, 1)
		prec1, prec5 = Helper.accuracy(out_hat, labels, topk=(1, 5))
		lossf.update(fnet.loss_G.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		if batch_idx % args.log_steps == 0:
			losses = fnet.get_current_losses()
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f D-real: %.4f D-fake: %.3f G-GAN: %.4f G-L1 %.3f G-D: %.4f Acc@1: %.3f Acc@5: %.3f\n'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch_size)+1, lossf.avg, losses['D_real'], losses['D_fake'], losses['G_GAN'], losses['G_L1'], losses['G_distill'],
							top1.avg, top5.avg))
			sys.stdout.flush()

	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))
	writer.add_scalar('L1 loss', losses['G_L1'], epoch)
	writer.add_scalar('GAN loss', losses['G_GAN'], epoch)
	writer.add_scalar('D-real', losses['D_real'], epoch)
	writer.add_scalar('D-fake', losses['D_fake'], epoch)
	writer.add_scalar('G-distill', losses['G_distill'], epoch)

	fnet.update_learning_rate()
	if (epoch - 1) % args.save_steps == 0 or epoch == init_epoch + args.epochs:
		Tester.evaluate_fgan(inet, fnet.netG, testloader, args.nactors)
		print('saving the latest fnet (epoch %d, total_iters %d)' % (epoch, total_steps))
		fnet.save_networks(epoch)

########################## Store ##################
print('Done')
