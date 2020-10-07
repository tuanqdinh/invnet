"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

from init_invnet import *
from torch.utils.tensorboard import SummaryWriter

args.inet_name = "inet_reg_{}_{}".format(args.dataset, args.name)
args.inet_save_dir = os.path.join(checkpoint_dir, args.inet_name)
Helper.try_make_dir(args.inet_save_dir)

writer = SummaryWriter('../results/runs/' + args.inet_name)
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
	save_filename = '%s_net.pth' % (args.resume)
	inet_path = os.path.join(args.inet_save_dir, save_filename)
	if os.path.isfile(inet_path):
		print("-- Loading checkpoint '{}'".format(inet_path))
		inet.load_state_dict(torch.load(inet_path))
	else:
		print("--- No checkpoint found at '{}'".format(inet_path))
		sys.exit('Done')
else:
	print('Load base model')
	inet_path = os.path.join(args.inet_save_dir, 'inet_{}_base.pth'.format(args.dataset))
	inet.load_state_dict(torch.load(inet_path))

in_shapes = inet.module.get_in_shapes()
if args.optimizer == "adam":
	optimizer = optim.Adam(inet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	optimizer = optim.SGD(inet.parameters(), lr=args.lr,
						  momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
##### Analysis ######
if analyse(args, inet, in_shapes, trainloader, testloader):
	sys.exit('Done')

##### Training ######
print('|  Train: mixup: {} mixup_hidden {} alpha {} epochs {}'.format(args.mixup, args.mixup_hidden, args.mixup_alpha, args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
test_objective = -np.inf
init_epoch = int(args.resume)
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

		logits, z, reweighted_target = inet(inputs, targets=targets, mixup_hidden=args.mixup_hidden, mixup=args.mixup, mixup_alpha=args.mixup_alpha)
		norms = torch.norm(z.view(z.shape[0], -1), dim=1)
		l_norm = (torch.max(norms) - torch.min(norms))/1024
		l_bce = bce_loss(softmax(logits), reweighted_target)
		loss = args.lamb * l_bce + l_norm

		# measure accuracy and record loss
		_, labels = torch.max(reweighted_target.data, 1)
		prec1, prec5 = Helper.accuracy(logits, labels, topk=(1, 5))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_steps == 0:
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f l_norm: %.4f  l_bce: %.4f Acc@1: %.3f Acc@5: %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch_size)+1, loss.data.item(), l_norm.data.item(), l_bce.data.item(),
							top1.avg, top5.avg))
			sys.stdout.flush()

	# scheduler.step()
	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))

	writer.add_scalar('Norm', l_norm, epoch)
	writer.add_scalar('BCE', l_bce, epoch)
	writer.add_scalar('loss', loss, epoch)

	if (epoch - 1) % args.save_steps == 0 or epoch == init_epoch + args.epochs:
		# evaluate
		print("Evaluating on Testset")
		Tester.evaluate(inet, testloader)
		#save
		Helper.save_networks(inet, args.inet_save_dir, epoch)

print('Done')
