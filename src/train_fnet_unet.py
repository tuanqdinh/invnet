"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_fusionnet import *
from fusionnet.unet import UnetModel
from tensorboardX import SummaryWriter
from ops import evaluate_unet
import time

writer = SummaryWriter('../results/runs/' + args.fnet_name)

#defaults
args.dataset_mode='aligned'
args.gpu_ids = [0]
args.pool_size=0

fnet = UnetModel(args)
fnet.setup(args)              # regular setup: load and print networks; create schedulers

init_epoch = int(args.resume_g)
if init_epoch > 0:
	fnet.load_networks(epoch=init_epoch)

if args.flag_overfit:
	print('Train to overfit using test set')
	# data_loader = test_loader
	data_loader = DataLoader(dataset=test_dataset,
		num_workers=args.threads,
		batch_size=args.batch_size, shuffle=False)
else:
	data_loader = train_loader

criterionL1 = torch.nn.L1Loss()
elapsed_time = 0
total_steps = len(iter(data_loader))
for epoch in range(1 + init_epoch, args.epochs + 1 + init_epoch):
	start_time = time.time()
	lr = Helper.learning_rate(args.lr, epoch - init_epoch)
	Helper.update_lr(fnet.optimizer_G, lr)
	print('|Learning rate = %.7f' % fnet.optimizers[0].param_groups[0]['lr'])
	for batch_idx, (images, targets) in enumerate(data_loader):
		inputs = images.to(device)
		targets = targets.to(device)
		N, A, C, H, W = inputs.shape
		# inputs = inputs[:, torch.randperm(A), ...]
		inputs = inputs.view(N, C*A, H, W)

		# forward
		fnet.set_input(inputs, targets)         # unpack data from dataset and apply preprocessing
		fnet.optimize_parameters()   # calculate loss functions, get gradients, update network weights

		if batch_idx % args.log_steps == 0:
			losses = fnet.get_current_losses()
			sys.stdout.write('\r')
			sys.stdout.write("===> Epoch[{}]({}/{}): log-cosh: {:.4f} L1 {:.4f}".format(epoch, batch_idx, total_steps, losses['logcosh'], losses['G_L1']))
			# print("===> Epoch[{}]({}/{}): G-L2 {:.4f}".format(epoch, batch_idx, total_steps, losses['G_L2']))
			sys.stdout.flush()
		if args.flag_overfit:
			break

	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))
	writer.add_scalar('L1 loss', losses['G_L1'], epoch)
	# writer.add_scalar('L2 loss', losses['G_L2'], epoch)
	#checkpoint
	if (epoch - 1) % args.save_steps == 0 or epoch == args.epochs + init_epoch:
		print('saving the latest fnet (epoch %d, total_iters %d)' % (epoch, total_steps))
		fnet.save_networks(epoch)
		# lossf, corrects = evaluate_unet(test_loader, fnet, inet, criterionL1, device)
		# print('Evaluate L1 {:.4f} Match: {:.4f}'.format(lossf, corrects))

	# fnet.update_learning_rate()                     # update learning rates at the end of every epoch.
print('Done')
