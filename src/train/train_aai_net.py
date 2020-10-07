"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""

from init import *
from models.critic import Discriminator

model = iResNet(nBlocks=args.nBlocks,
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


# init actnrom parameters
init_batch = get_init_batch(trainloader, args.init_batch).to(device)
print("initializing actnorm parameters...")
with torch.no_grad():
	model(init_batch, ignore_logdet=True)
print("initialized")

use_cuda = torch.cuda.is_available()
model.cuda()
# TODO:
model = torch.nn.DataParallel(model, range(torch.cuda.device_count()))
cudnn.benchmark = True
in_shapes = model.module.get_in_shapes()

netD = Discriminator().to(device)
netD.apply(Helper.weights_init)

if args.optimizer == "adam":
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == "adamax":
	optimizer = optim.Adamax(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
else:
	optimizer = optim.SGD(model.parameters(), lr=args.lr,
						  momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)

optimizer_D = optim.Adam(netD.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# log
# setup logging with visdom
viz = visdom.Visdom(port=args.vis_port, server="http://" + args.vis_server)
assert viz.check_connection(), "Could not make visdom"

with open(param_path, 'w') as f:
	f.write(json.dumps(args.__dict__))
train_log = open(trainlog_path, 'w')
try_make_dir(args.save_dir)
try_make_dir(sample_path)

########################## Training ##################
print('|  Train Epochs: ' + str(args.epochs))
print('|  Initial Learning Rate: ' + str(args.lr))

elapsed_time = 0
test_objective = -np.inf
epoch = 0
for epoch in range(1, 1+args.epochs):
	start_time = time.time()

	model.train()
	correct = 0
	total = 0
	reg = 0
	# update lr for this epoch (for classification only)
	lr = learning_rate(args.lr, epoch)
	update_lr(optimizer, lr)

	for batch_idx, (inputs, targets) in enumerate(trainloader):
		cur_iter = (epoch - 1) * len(trainloader) + batch_idx
		# if first epoch use warmup
		if epoch - 1 <= args.warmup_epochs:
			this_lr = args.lr * float(cur_iter) / (args.warmup_epochs * len(trainloader))
			update_lr(optimizer, this_lr)
			update_lr(optimizer_D, this_lr)

		inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
		inputs = Variable(inputs, requires_grad=True)
		targets = Variable(targets)
		bs = inputs.shape[0] // 2
		if inputs.shape[0] % 2 == 1:
			inputs = inputs[1:, ...]
		input_1 = inputs[:bs, ...]
		input_2 = inputs[bs:, ...]


		if batch_idx % 2 == 0:
			# deactivate critics
			Helper.deactivate(netD.parameters())
			Helper.activate(model.parameters())
			optimizer.zero_grad()
			out, out_bij = model(inputs)
			loss = criterion(out, targets) # Loss
			# modelling
			z_1 = out_bij[:bs, ...]
			z_2 = out_bij[bs:, ...]
			alpha = torch.rand(bs, 1, 1, 1, requires_grad=False).to(device)
			alphas = alpha.expand(z_1.shape)* 0.5
			z_3 = alphas * z_1 + (1 - alphas) * z_2
			x_3 = model.module.inverse(z_3)
			v = netD(x_3)
			loss_c = (v * v).mean()
			loss = loss + 0.5 * loss_c
			loss.backward()  # Backward Propagation
			optimizer.step()  # Optimizer update
		else:
			Helper.deactivate(model.parameters())
			Helper.activate(netD.parameters())
			optimizer_D.zero_grad()
			with torch.no_grad():
				out, out_bij = model(inputs)
				z_1 = out_bij[:bs, ...]
				z_2 = out_bij[bs:, ...]
				alpha = torch.rand(bs, 1, 1, 1, requires_grad=False).to(device)
				alphas = alpha.expand(z_1.shape)* 0.5
				z_3 = alphas * z_1 + (1 - alphas) * z_2
				x_3 = model.module.inverse(z_3)
			v = netD(x_3)
			v_gamma = netD(inputs)
			loss_c = criterionMSE(v, alpha.view(bs))
			loss_gamma = (v_gamma * v_gamma).mean()
			loss_c = loss_c + loss_gamma
			loss_c.backward()  # Backward Propagation
			optimizer_D.step()  # Optimizer update

			loss = criterion(out, targets) # Loss


		_, predicted = torch.max(out.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum()
		if batch_idx % 1 == 0:
			sys.stdout.write('\r')
			sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Reg: %.4f  Acc@1: %.3f'
						 % (epoch, args.epochs, batch_idx+1,
							(len(trainset)//args.batch)+1, loss.data.item(), loss_c.data.item(),
							100.*correct.type(torch.FloatTensor)/float(total)))
			sys.stdout.flush()

	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))
	# test inverse function
	if epoch % 10 == 0:
		eval_invertibility(model, testloader, epoch)
		net_path = os.path.join(checkpoint_path, '{}_e{}.pkt'.format(args.model_name, epoch))
		print('Save checkpoint at ', net_path)
		state = {
			'model': model,
			'objective': test_objective,
			'epoch': epoch,
		}
		torch.save(state, net_path)

########################## Test ##################
if args.flag_test:
	print('Testing model')
	test_log = open(testlog_path, 'w')
	test_objective = test(test_objective, args, model, epoch, testloader, viz, use_cuda, test_log)
	print('* Test results : objective = %.2f%%' % (test_objective))
	with open(final_path, 'w') as f:
		f.write(str(test_objective))

########################## Store ##################
net_path = os.path.join(checkpoint_path, '{}_e{}.pkt'.format(args.model_name, args.epochs))
print('Save checkpoint at ', net_path)
state = {
	'model': model,
	'objective': test_objective,
	'epoch': epoch,
}
torch.save(state, net_path)

eval_invertibility(model, testloader, num_epochs=args.epochs)
