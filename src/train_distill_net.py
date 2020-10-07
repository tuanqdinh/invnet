"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

from init_invnet import *
from invnet.ops.mixup_ops import to_one_hot
from invnet.resnet import ResNet, BasicBlock
from ops import load_inet, load_networks, loss_fn_kd, evaluate_dnet
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../results/runs/' + args.fnet_name)

########################## Load ##################
isvalid, inet = load_inet(args, device)
if not(isvalid):
	sys.exit('Invalid inet')

fnet = ResNet(BasicBlock, [7, 7, 7], input_nc=args.nactors*args.input_nc).to(device)
init_epoch = int(args.resume_g)
if init_epoch > 0:
	isvalid, fnet = load_networks(fnet, args.resume_g, args.fnet_save_dir)
	if not(isvalid):
		sys.exit('Invalid fnet')

########################## Training ###################################
optimizer = optim.Adam(fnet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)

print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))
elapsed_time = 0
total_steps = len(iter(trainloader))
A = args.nactors
for epoch in range(init_epoch + 1, init_epoch + 1 + args.epochs):
	inet.eval()
	start_time = time.time()
	lossf = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	# update lr for this epoch (for classification only)
	lr = Helper.learning_rate(args.lr, epoch - init_epoch)
	Helper.update_lr(optimizer, lr)
	lr = optimizer.param_groups[0]['lr']
	print('|Learning Rate: ' + str(lr))
	for batch_idx, (inputs, labels) in enumerate(trainloader):
		cur_iter = (epoch - 1) * len(trainloader) + batch_idx
		# if first epoch use warmup
		if epoch - 1 <= args.warmup_epochs:
			this_lr = args.lr * float(cur_iter) / (args.warmup_epochs * len(trainloader))
			Helper.update_lr(optimizer, this_lr)

		inputs = inputs.to(device)
		labels = labels.to(device)
		N, C, H, W = inputs.shape
		M = N // A
		N = M * A
		inputs = inputs[:N, ...]
		labels = labels[:N, ...]

		logits, _, _ = inet(inputs)
		logits_mean = logits.view(M, A, args.nClasses).mean(dim=1)

		# fusion inputs
		x_g = inputs.view(M, A, C, H, W).view(M, A*C, H, W)
		logits_mean_g, _, _ = fnet(x_g)

		# inet loss
		labels_mean = to_one_hot(labels, args.nClasses).view(M, A, args.nClasses).mean(dim=1)
		optimizer.zero_grad()
		loss = loss_fn_kd(logits_mean_g, labels_mean, logits_mean, alpha=0.1, temperature=6)
		loss.backward()
		optimizer.step()

		# measure accuracy and record loss
		_, labels = torch.max(labels_mean.data, 1)
		prec1, prec5 = Helper.accuracy(logits_mean_g, labels, topk=(1, 5))
		lossf.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		top5.update(prec5.item(), inputs.size(0))

		if batch_idx % args.log_steps == 0:
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f Acc@1: %.3f Acc@5: %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch_size)+1, lossf.avg,
							top1.avg, top5.avg))
			sys.stdout.flush()

	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))
	writer.add_scalar('train_loss', loss, epoch)
	# scheduler.step()

	if (epoch - 1) % args.save_steps == 0 or epoch == init_epoch + args.epochs:
		# evaluate
		corrects = evaluate_dnet(testloader, fnet, inet, device, args.nactors, args.nClasses)
		print("Evaluating on Testset: Acc@1: {:.4f}".format(corrects))
		#save
		Helper.save_networks(fnet, args.fnet_save_dir, epoch)

########################## Store ##################
print('Done')
