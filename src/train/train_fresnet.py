"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_invnet import *
from init_fusionnet import *
from invnet.fresnet import conv_iResNet as fResNet
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('../results/runs/fashion_mnist_experiment_1')

fnet = fResNet(nBlocks=args.nBlocks,
				nStrides=args.nStrides,
				nChannels=args.nChannels,
				nClasses=10,
				init_ds=args.init_ds,
				inj_pad=args.inj_pad,
				in_shape=in_shape,
				nactors=args.nactors,
				input_nc=args.input_nc,
				coeff=args.coeff,
				numTraceSamples=args.numTraceSamples,
				numSeriesTerms=args.numSeriesTerms,
				n_power_iter = args.powerIterSpectralNorm,
				density_estimation=args.densityEstimation,
				actnorm=(not args.noActnorm),
				learn_prior=(not args.fixedPrior),
				nonlin=args.nonlin).to(device)


fnet = torch.nn.DataParallel(fnet, range(torch.cuda.device_count()))

if os.path.isfile(fnet_path):
	print("-- Loading checkpoint '{}'".format(fnet_path))
	# fnet = torch.load(fnet_path)
	fnet.load_state_dict(torch.load(fnet_path))
else:
	print("No checkpoint found at: ", fnet_path)

optim_fnet = optim.Adam(fnet.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
# optim_fnet = optim.SGD(fnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
fnet_scheduler = Helper.get_scheduler(optim_fnet, args)

# Training
total_steps = len(iter(train_loader))
num_epochs = args.niter + args.niter_decay
for epoch in range(1, num_epochs + 1):
	for batch_idx, batch_data in enumerate(train_loader):
		batch_data = batch_data.to(device)
		# [-2.7; 2.3]
		inputs = batch_data[:, :-1, ...]
		inputs = inputs[:, torch.randperm(args.nactors), ...]
		N, A, C, H, W = inputs.shape
		inputs = inputs.view(N, A * C, H, W)
		# if 1 == ninputs:
		# 	# concat input
		# 	inputs = inputs.permute(0, 2, 1, 3, 4)
		# [-30; 30]
		targets = batch_data[:, -1, ...]
		optim_fnet.zero_grad()
		# [-2.3, 2.3]
		preds = fnet(inputs).view(targets.shape)
		loss = criterionMSE(preds, targets)/args.batch_size
		loss.backward()
		optim_fnet.step()

		if batch_idx % args.log_steps == 0:
			print("===> Epoch[{}]({}/{}): L1-pixel: {:.4f}".format(
			epoch, batch_idx, total_steps, loss.item()))

	writer.add_scalar('training loss', loss.item(), epoch)
	Helper.update_learning_rate(fnet_scheduler, optim_fnet)
	#checkpoint
	if (epoch - 1) % args.save_steps == 0:
		print("Saving")
		fnet_path = os.path.join(checkpoint_dir, '{}/{}_{}_e{}.pth'.format(root, root, args.model_name, args.resume_g + epoch))
		torch.save(fnet.state_dict(), fnet_path)
		# Helper.save_images(targets, sample_dir, args.model_name, 'target', epoch)

		# Helper.save_images(preds, sample_dir, args.model_name, 'pred', epoch)
		# evaluate classification
		# Tester.evaluate_fnet(inet, fnet, dataloader, stats)

# final model
fnet_path = os.path.join(checkpoint_dir, '{}/{}_{}_e{}.pth'.format(root, root, args.model_name, args.resume_g + epoch))
torch.save(fnet.state_dict(), fnet_path)
writer.close()
print('Done')
