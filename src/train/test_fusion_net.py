"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_fusionnet import *
# define fnet
if args.concat_input:
	ninputs = 1
	args.input_nc = args.input_nc * args.nactors
else:
	ninputs = args.nactors

fnet = define_G(
		input_nc=args.input_nc,
		output_nc=args.output_nc,
		nactors=ninputs,
		ngf=args.ngf,
		norm='batch',
		use_dropout=False,
		init_type='normal',
		init_gain=0.02,
		gpu_id=device)

# fnet = torch.nn.DataParallel(fnet, range(torch.cuda.device_count()))
scores = []
for i in range(11):
	e = i * 10 + 1
	if e > 100:
		e = 100
	fnet_path = os.path.join(checkpoint_dir, '{}/{}_{}_e{}.pth'.format(root, root, args.model_name, e))
	fnet.load_state_dict(torch.load(fnet_path))
	lossf = []
	for batch_idx, batch_data in enumerate(train_loader):
		batch_data = batch_data.to(device)
		# [-2.7; 2.3]
		inputs = batch_data[:, :-1, ...]
		inputs = inputs[:, torch.randperm(args.nactors), ...]

		targets = batch_data[:, -1, ...]
		preds = fnet(inputs)
		loss = criterionMSE(preds, targets) / args.batch_size
		lossf.append(loss.item())
	print('Epoch {}: {:.4f}'.format(e, np.mean(lossf)))
	scores.append(np.mean(lossf))

print(scores)
print('Done')
