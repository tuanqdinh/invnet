"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_fusionnet import *
from fusionnet.pix2pix import Pix2PixModel
from ops import load_inet, evaluate_unet

# iResNet data
isvalid, inet = load_inet(args, device)
if not(isvalid):
	sys.exit('Invalid inet')

# Fusion Net
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids=[0]
args.pool_size=0
args.lambda_L1 = args.lamb

fnet = Pix2PixModel(args)
init_epoch = int(args.resume_g)
fnet.load_networks(epoch=init_epoch)
# Training

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
print('L1 {:.4f} Acc1: {:.4f} Acc2: {:.4f}'.format(lossf, corrects[0], corrects[1]))
