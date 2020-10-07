"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_invnet import *
from init_fusionnet import *
from invnet.fresnet import conv_iResNet as fResNet

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
scores = []
for i in range(11):
	e = i * 10 + 1
	if e > 100:
		e = 100
	fnet_path = os.path.join(checkpoint_dir, '{}/{}_{}_e{}.pth'.format(root, root, args.model_name, e))
	fnet.load_state_dict(torch.load(fnet_path))
	lossf = []
	for batch_idx, batch_data in enumerate(test_loader):
		batch_data = batch_data.to(device)
		# [-2.7; 2.3]
		inputs = batch_data[:, :-1, ...]
		inputs = inputs[:, torch.randperm(args.nactors), ...]
		N, A, C, H, W = inputs.shape
		inputs = inputs.view(N, A * C, H, W)
		targets = batch_data[:, -1, ...]
		preds = fnet(inputs).view(targets.shape)
		loss = criterionMSE(preds, targets)/args.batch_size
		lossf.append(loss.item())

	print('Epoch {}: {:.4f}'.format(e, np.mean(lossf)))
	scores.append(np.mean(lossf))

print(scores)
print('Done')
