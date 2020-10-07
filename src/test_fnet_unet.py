"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_fusionnet import *
from fusionnet.unet import UnetModel
from ops import load_inet, evaluate_unet

# iResNet data
isvalid, inet = load_inet(args, device)
if not(isvalid):
	sys.exit('Invalid inet')

# Fusion Net
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
args.pool_size=0
fnet = UnetModel(args)
fnet.setup(args)
init_epoch = int(args.resume_g)
fnet.load_networks(epoch=init_epoch)

if args.flag_plot:
	from utils.tester import Tester
	from utils.provider import Provider
	trainset, testset, in_shape = Provider.load_data(args.dataset, args.data_dir)
	trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
	testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

	Tester.plot_reconstruction(args.fnet_name + '_train', '../results/plots', inet, fnet.netG, trainloader, args.nactors)
	Tester.plot_reconstruction(args.fnet_name + '_test', '../results/plots', inet, fnet.netG, testloader, args.nactors)

	sys.exit('Plotted')


if args.flag_test:
	print('Evaluate on Test Set')
	data_loader = test_loader
elif args.flag_val:
	print('Evaluate on Validation Set')
	data_loader = val_loader
else:
	print('Evaluate on Train Set')
	data_loader = train_loader

criterion = torch.nn.L1Loss()
lossf, corrects = evaluate_unet(data_loader, fnet, inet, criterion, device)
print('L1 {:.4f} Match: {:.4f}'.format(lossf, corrects))
