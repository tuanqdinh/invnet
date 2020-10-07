"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

from init_invnet import *
from ops import load_networks
from tensorboardX import SummaryWriter

map_targets = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]

def convert_to_super(targets):
	m = torch.tensor(map_targets, dtype=torch.float32).cuda()
	m = m.unsqueeze(0).expand(targets.shape)
	x = m * targets
	x = x.sum(dim=1)
	y = 1 - x
	r = torch.cat([y.unsqueeze(1), x.unsqueeze(1)], dim=1)
	return r

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

# viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
# assert viz.check_connection(), "Could not make visdom"

if args.resume > -1:
	isvalid, inet = load_networks(inet, args.resume, args.inet_save_dir)
	init_epoch = int(args.resume)
	if not(isvalid):
		sys.exit('Done')
else:
	init_epoch = 0

####################### Analysis ##############################
in_shapes = inet.module.get_in_shapes()
if analyse(args, inet, in_shapes, trainloader, testloader):
	sys.exit('Done')

####################### Training ##############################
writer_train = SummaryWriter('../results/runs/' + args.inet_name + '_train')
writer_val = SummaryWriter('../results/runs/' + args.inet_name + '_val')

optimizer = optim.SGD(inet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

vTester = iTester(dataloader=valloader)

print('Model: ', args.iname)
print(inet)
print('|  Train: mixup: {} mixup_hidden {} alpha {} epochs {}'.format(args.mixup, args.mixup_hidden, args.mixup_alpha, args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
best_accuracy = -np.inf

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
		super_targets = np.asarray([map_targets[x] for x in targets])
		targets = Variable(targets).cuda()
		super_targets = Variable(torch.tensor(super_targets, dtype=torch.long)).cuda()

		if args.mixup or args.mixup_hidden:
			logits, z, reweighted_targets = inet(inputs, targets=targets, mixup_hidden=args.mixup_hidden, mixup=args.mixup, mixup_alpha=args.mixup_alpha)
			super_reweighted_targets = convert_to_super(reweighted_targets)
			fine_loss = bce_loss(softmax(logits[0]), reweighted_targets)
			super_loss = bce_loss(softmax(logits[1]), super_reweighted_targets)
			_, fine_labels = torch.max(reweighted_targets.data, 1)
			_, super_labels = torch.max(super_reweighted_targets.data, 1)
			prec1 = Helper.accuracy(logits[0], fine_labels, topk=(1,))
			prec5 = Helper.accuracy(logits[1], super_labels, topk=(1,))
			loss = fine_loss + 0.5 * super_loss
		else:
			logits, z, _ = inet(inputs)
			fine_loss = criterionCE(logits[0], targets)
			super_loss = criterionCE(logits[1], super_targets)
			prec1 = Helper.accuracy(logits[0], targets, topk=(1,))
			prec5 = Helper.accuracy(logits[1], super_targets, topk=(1,))
			loss = fine_loss + 0.5 * super_loss

		# measure accuracy and record loss
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1[0].item(), inputs.size(0))
		top5.update(prec5[0].item(), inputs.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		if batch_idx % args.log_steps == 0:
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f Acc@1: %.3f Acc@5: %.3f \tZ: %.3f %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch_size)+1, losses.avg,
							top1.avg, top5.avg, z.median(), z.abs().max()))
			sys.stdout.flush()

	# scheduler.step()
	writer_train.add_scalar('loss', loss, epoch)
	writer_train.add_scalar('accuracy-class', top1.avg, epoch)
	writer_train.add_scalar('accuracy-super', top5.avg, epoch)

	# writer_val.add_scalar('loss', l, epoch)
	# writer_val.add_scalar('accuracy', acc1, epoch)

	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))

	if (epoch - 1) % args.save_steps == 0 or epoch == init_epoch + args.epochs:
		#save
		l, acc1, acc5 = vTester.evaluate(inet)
		sys.stdout.write('Evaluating on Val-set: Loss: %.3f Acc@1: %.3f Acc@5: %.3f' % (l, acc1, acc5))

		if acc1 > best_accuracy:
			best_accuracy = acc1
			is_best = True
		else:
			is_best = False
		Helper.save_networks(inet, args.inet_save_dir, epoch, args, loss, prec1, l, acc1, is_best)

########################## Store ##################
print('Done')
