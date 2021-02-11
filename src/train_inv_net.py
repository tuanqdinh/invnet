"""
	@author xxx xxx@cs.xxx.edu
	@date 02/14/2020
"""

from init_invnet import *
from ops import load_networks
# from tensorboardX import SummaryWriter

map_targets = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]

def convert_to_super(targets):
	m = torch.tensor(map_targets, dtype=torch.float32).cuda()
	m = m.unsqueeze(0).expand(targets.shape)
	x = m * targets
	x = x.sum(dim=1)
	y = 1 - x
	r = torch.cat([y.unsqueeze(1), x.unsqueeze(1)], dim=1)
	return r

args.nChannels = [2, 8, 32]
args.bottleneck_mult = 1
args.init_ds = 2
inet = iResNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
				nChannels=args.nChannels, nClasses=10,
				init_ds=args.init_ds, dropout_rate=0.1, affineBN=True,
				in_shape=in_shape, mult=args.bottleneck_mult, actnorm=not(args.noActnorm)).to(device)


init_batch = Helper.get_init_batch(trainloader, args.init_batch)
print("initializing actnorm parameters...")
with torch.no_grad():
	inet(init_batch.to(device))
print("initialized")

inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))

if args.resume > 0:
	isvalid, inet = load_networks(inet, args.resume, args.inet_save_dir)
	init_epoch = int(args.resume)
	if not(isvalid):
		sys.exit('Done')
else:
	init_epoch = 0

####################### Analysis ##############################
if analyse(args, inet, None, trainloader, testloader):
	sys.exit('Done')

####################### Training ##############################
# writer_train = SummaryWriter('../results/runs/' + args.inet_name + '_train')
# writer_val = SummaryWriter('../results/runs/' + args.inet_name + '_val')

# optimizer = optim.SGD(inet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

optimizer = optim.Adam(inet.parameters(), lr=args.lr)

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
		targets = Variable(targets).cuda()

		if args.mixup or args.mixup_hidden:
			logits, z, reweighted_targets = inet(inputs, targets=targets, mixup_hidden=args.mixup_hidden, mixup=args.mixup, mixup_alpha=args.mixup_alpha)
			loss = bce_loss(softmax(logits), reweighted_targets)
			_, labels = torch.max(reweighted_targets.data, 1)
			prec1, prec5 = Helper.accuracy(logits, labels, topk=(1,5))
		else:
			logits, z, _ = inet(inputs)
			loss = criterionCE(logits, targets)
			prec1, prec5 = Helper.accuracy(logits, targets, topk=(1,5))

		# measure accuracy and record loss
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

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
			# inet.eval()
			# z = inet.module.embed(inputs)
			# x_inversed = inet.module.inverse(z)
			# Helper.save_images(x_inversed, '../results/samples/', 'revnet', 'inversed', epoch)
			# Helper.save_images(inputs, '../results/samples/', 'revnet', 'real', epoch)
			# from IPython import embed; embed()

	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))

	if (epoch - 1) % args.save_steps == 0 or epoch == init_epoch + args.epochs:
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
