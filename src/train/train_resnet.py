"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

from init_invnet import *
from invnet.resnet import ResNet, BasicBlock

if args.input_nc == 1:
	nblocks = [5, 5, 5]
else:
	nblocks = [9, 9, 9]

inet = torch.nn.DataParallel(ResNet(BasicBlock, nblocks, input_nc=args.input_nc)).to(device)

if os.path.isfile(inet_path):
	print("-- Loading checkpoint '{}'".format(inet_path))
	inet.load_state_dict(torch.load(inet_path))
else:
	print("--- No checkpoint found at '{}'".format(inet_path))
	args.resume = 0

if args.optimizer == "adam":
	optimizer = optim.Adam(inet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	optimizer = optim.SGD(inet.parameters(), lr=args.lr,
						  momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
##### Analysis ######
if analyse(args, inet, in_shape, trainloader, testloader):
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

		logits, _, _ = inet(inputs)
		loss = criterionCE(logits, targets)

		prec1, prec5 = Helper.accuracy(logits, targets, topk=(1, 5))
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

	if (epoch - 1) % args.save_steps == 0:
		model_name = '{}_{}'.format(root, args.model_name)
		Helper.save_checkpoint(inet, test_objective, checkpoint_dir, model_name, epoch)

########################## Store ##################
Helper.save_checkpoint(inet, test_objective, checkpoint_dir, args.model_name, args.epochs)
print('Done')
