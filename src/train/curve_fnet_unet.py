"""
	@author Tuan Dinh tuandinh@cs.wisc.edu
	@date 02/14/2020
"""
from init_fusionnet import *
from fusionnet.unet import UnetModel
from ops import load_inet, evaluate_unet
import time
from utils.plotter import Plotter

save_name = '../results/curves_' + args.fnet_name + '.npy'

if args.flag_plot:
	if os.path.isfile(save_name):
		scores = np.load(save_name)
		# scores[:, :, 0] = scores[:, :, 0]**2
		x = np.arange(args.niter) * 10
		Plotter.plot_curves(scores, x, 0, 'Loss learning curve', 'Loss', '../results/plots/curves_' + args.fnet_name + '_loss.png')
		Plotter.plot_curves(scores, x, 1, 'Match learning curve', 'Match', '../results/plots/curves_' + args.fnet_name + '_match.png')
	else:
		print('File not exist')
	sys.exit('Done')
isvalid, inet = load_inet(args, device)
if not(isvalid):
	sys.exit('Invalid inet')

inet.eval()
# Fusion Net
args.norm='batch'
args.dataset_mode='aligned'
args.gpu_ids = [0]
args.pool_size=0
fnet = UnetModel(args)
fnet.setup(args)

# criterion = torch.nn.L1Loss()
criterion = torch.nn.MSELoss()

scores = np.zeros((args.niter, 3, 2))
# abuse
elapsed_time=0
start_time = time.time()
for offset in range(args.niter):
	resume_g = offset * 10 + 1
	fnet.load_networks(epoch=resume_g)
	fnet.eval()

	lossf, corrects = evaluate_unet(test_loader, fnet, inet, criterion, device)
	scores[offset, 2, 0] = lossf
	scores[offset, 2, 1] = corrects

	lossf, corrects = evaluate_unet(train_loader, fnet, inet, criterion, device)
	scores[offset, 0, 0] = lossf
	scores[offset, 0, 1] = corrects

	lossf, corrects = evaluate_unet(val_loader, fnet, inet, criterion, device)
	scores[offset, 1, 0] = lossf
	scores[offset, 1, 1] = corrects

	print('|{} - L1 Train {:.4f} Val {:.4f} Test {:.4f}'.format(offset, scores[offset, 0, 0], scores[offset, 1, 0], scores[offset, 2, 0]))
	print('|{} - M  Train {:.4f} Val {:.4f} Test {:.4f}'.format(offset, scores[offset, 0, 1], scores[offset, 1, 1], scores[offset, 2, 1]))
	epoch_time = time.time() - start_time
	elapsed_time += epoch_time
	print('| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))

np.save(save_name, scores)
x = np.arange(args.niter) * 10
Plotter.plot_curves(scores, x, 0, 'Loss learning curve', 'Loss', '../results/plots/curves_' + args.fnet_name + '_loss.png')
Plotter.plot_curves(scores, x, 1, 'Match learning curve', 'Match', '../results/plots/curves_' + args.fnet_name + '_match.png')
