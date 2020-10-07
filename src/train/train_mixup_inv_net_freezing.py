"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

from init_invnet import *

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

inet_path = os.path.join(checkpoint_dir, 'inet/inet_{}_freezing_e{}.pth'.format(args.inet_name, args.resume))
if os.path.isfile(inet_path):
	print("-- Loading checkpoint '{}'".format(inet_path))
	"""
	checkpoint = torch.load(inet_path)
	# inet = checkpoint['inet']
	best_inet = checkpoint['inet']
	# load dict only
	inet.load_state_dict(best_inet.state_dict())
	best_objective = checkpoint['objective']
	print('----- Objective: {:.4f}'.format(best_objective))
	"""
	inet.load_state_dict(torch.load(inet_path))
else:
	print("--- No checkpoint found at '{}'".format(inet_path))
	args.resume = 0


in_shapes = inet.module.get_in_shapes()
##### Analysis ######
if analyse(args, inet, in_shapes, trainloader, testloader):
	sys.exit('Done')

from fusionnet.pix2pix import Pix2PixModel
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
#train
args.pool_size=0
args.gan_mode='lsgan'
args.netD = 'pixel'
args.fnet_name = "{}_{}_{}_{}".format(args.dataset, 'large', args.nactors, args.gan_mode)
args.isTrain = False

pix2pix = Pix2PixModel(args)
pix2pix.setup(args)              # regular setup: load and print networks; create schedulers
pix2pix.load_networks(epoch=int(args.resume_g))
fnet = pix2pix.netG
fnet.eval()

def atanh(x):
	return 0.5*torch.log((1+x)/(1-x))

def set_requires_grad(nets, requires_grad=False):
	"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
	Parameters:
		nets (network list)   -- a list of networks
		requires_grad (bool)  -- whether the networks require gradients or not
	"""
	if not isinstance(nets, list):
		nets = [nets]
	for net in nets:
		if net is not None:
			for param in net.parameters():
				param.requires_grad = requires_grad


if args.optimizer == "adam":
	optimizer = optim.Adam(inet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	optimizer = optim.SGD(inet.parameters(), lr=args.lr,
						  momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Freezing
set_requires_grad(inet.module.stack, False)


##### Training ######
print('|  Train: mixup: {} mixup_hidden {} alpha {} epochs {}'.format(args.mixup, args.mixup_hidden, args.mixup_alpha, args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
test_objective = -np.inf
init_epoch = int(args.resume)
A = args.nactors
for epoch in range(init_epoch + 1, init_epoch + 1 + args.epochs):
	start_time = time.time()
	inet.train()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	# update lr for this epoch (for classification only)
	lr = Helper.learning_rate(args.lr, epoch - init_epoch)
	Helper.update_lr(optimizer, lr)
	print('|Learning Rate: ' + str(lr))
	for batch_idx, (inputs, targets) in enumerate(trainloader):
		cur_iter = (epoch - 1) * len(trainloader) + batch_idx
		# if first epoch use warmup
		if epoch - 1 <= args.warmup_epochs:
			this_lr = args.lr * float(cur_iter) / (args.warmup_epochs * len(trainloader))
			Helper.update_lr(optimizer, this_lr)

		inputs = Variable(inputs, requires_grad=True).cuda()
		targets = Variable(targets).cuda()

		# update inputs and target here
		N, C, H, W = inputs.shape
		M = N // A
		N = M * A
		inputs = inputs[:N, ...].view(M, A, C, H, W)
		targets = targets[:N, ...].view(M, A)

		# fusion inputs
		x_g = inputs.view(M, A*C, H, W)
		x_fused_g = atanh(fnet(x_g))

		# Z
		_, z, _ = inet(inputs.view(M*A, C, H, W))
		z = z.view((M, A, z.shape[1], z.shape[2], z.shape[3]))
		target_missed = targets[:, 0]

		z_missed = z[:, 0, ...]
		z_rest = z[:, 1:, ...].sum(dim=1)
		z_fused = z.mean(dim=1)
		_, z_fused_g, _ = inet(x_fused_g)
		z_missed_g = A * z_fused_g - z_rest
		logits = inet.module.classifier(z_missed_g)

		# x_missed_g = inet.module.inverse(z_missed_g)

		# training here
		# inputs = Variable(x_missed_g.data)
		# targets = target_missed
		# logits, _, reweighted_target = inet(inputs, targets=targets, mixup_hidden=args.mixup_hidden, mixup=args.mixup, mixup_alpha=args.mixup_alpha)
		# loss = bce_loss(softmax(logits), reweighted_target)
		loss = criterionCE(logits, target_missed)
		# measure accuracy and record loss
		# _, labels = torch.max(reweighted_target.data, 1)
		labels = target_missed
		prec1, prec5 = Helper.accuracy(logits, labels, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_steps == 0:
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f Acc@5: %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch_size)+1, loss.data.item(),
							top1.avg, top5.avg))
			sys.stdout.flush()

	# scheduler.step()
	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))

	if (epoch - 1) % args.save_steps == 0 or epoch == init_epoch + args.epochs:
		Tester.evaluate_fgan(inet, fnet, testloader, stats=None, nactors=args.nactors, nchannels=in_shape[0], concat_input=True)
		inet_path = os.path.join(checkpoint_dir, 'inet/inet_{}_freezing_e{}.pth'.format(args.inet_name, epoch))
		torch.save(inet.state_dict(), inet_path)

########################## Store ##################
print('Done')
