"""
"""
import os, math
import numpy as np
import torch
import torchvision
from torch.optim import lr_scheduler

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


class Helper:
	#### ==== Network Printout ==== ######
	# Calculate the gradient norm of parameters of model
	@staticmethod
	def get_grad_norm(model):
		return [torch.norm(p.grad).cpu().numpy().item() for p in model.parameters()]

	@staticmethod
	def get_norm(model):
		return [torch.norm(p.detach()).cpu().numpy().item() for p in model.parameters()]

	@staticmethod
	def get_grad_max(model):
		a = []
		try:
			for p in model.parameters():
				a += [(p.grad.max().item(), p.max().item())]
		except:
			a = a
		return a

	@staticmethod
	def get_num_params(model):
		num_params = sum(p.numel() for p in model.parameters())
		return num_params

	@staticmethod
	def weights_init(m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			m.weight.data.normal_(0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			m.weight.data.normal_(1.0, 0.02)
			m.bias.data.fill_(0)

	@staticmethod
	def get_init_batch(dataloader, batch_size):
		"""
		gets a batch to use for init
		"""
		batches = []
		seen = 0
		for x, y in dataloader:
			batches.append(x)
			seen += x.size(0)
			if seen >= batch_size:
				break
		batch = torch.cat(batches)
		return batch

	@staticmethod
	def deactivate(params):
		for p in params:
			p.requires_grad = False

	@staticmethod
	def activate(params):
		for p in params:
			p.requires_grad = True

	@staticmethod
	def try_make_dir(d):
		if not os.path.isdir(d):
			# os.mkdir(d)
			os.makedirs(d)

	@staticmethod
	def save_images(samples, sample_dir, model_name, sample_name, batch_idx):
		bs = samples.shape[0]
		torchvision.utils.save_image(samples.cpu(), os.path.join(sample_dir, "{}_{}_{}.jpg".format(
			model_name, sample_name, batch_idx)), int(bs**.5), normalize=True)

	@staticmethod
	def save_checkpoint(model, test_objective, args, checkpoint_path, model_name, epoch):
		net_path = os.path.join(checkpoint_path, '{}_e{}.pth'.format(model_name, epoch))
		print('Save checkpoint at ', net_path)
		state = {
			'model': model.state_dict(),
			'objective': test_objective,
			'opt': args
		}
		torch.save(state, net_path)

	@staticmethod
	def save_networks(net, save_dir, epoch, args, train_loss, train_acc1, val_loss, val_acc1, is_best):
		"""Save all the networks to the disk.

		Parameters:
			epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
		"""
		save_filename = '%s_net.pth' % (epoch)
		save_path = os.path.join(save_dir, save_filename)
		state = {
			'model': net.state_dict(),
			'loss': [val_loss, train_loss],
			'accuracy': [val_acc1, train_acc1],
			'opt': args,
			'epoch': epoch
		}
		torch.save(state, save_path)
		print('\nSave model ', save_filename)

		torch.save(state, os.path.join(save_dir, 'current_net.pth'))
		if is_best:
			best_path = os.path.join(save_dir, 'best_net.pth')
			torch.save(state, best_path)


	#iresnet
	@staticmethod
	def learning_rate(init, epoch, factor=40):
		optim_factor = 0
		if epoch > 3 * factor:
			optim_factor = 3
		elif epoch > 2 * factor:
			optim_factor = 2
		elif epoch > factor:
			optim_factor = 1
		# optim_factor = 0
		# if epoch > 160:
		# 	optim_factor = 3
		# elif epoch > 120:
		# 	optim_factor = 2
		# elif epoch > 60:
		# 	optim_factor = 1
		return init*math.pow(0.2, optim_factor)

	@staticmethod
	def update_lr(optimizer, lr):
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr

	# fusion net
	@staticmethod
	def get_scheduler(optimizer, opt):
		if opt.lr_policy == 'lambda':
			def lambda_rule(epoch):
				lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
				return lr_l
			scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
		elif opt.lr_policy == 'step':
			scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
		elif opt.lr_policy == 'plateau':
			scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
		elif opt.lr_policy == 'cosine':
			scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
		else:
			return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
		return scheduler

	@staticmethod
	# update learning rate (called once every epoch)
	def update_learning_rate(scheduler, optimizer):
		scheduler.step()
		lr = optimizer.param_groups[0]['lr']
		print('learning rate = %.7f' % lr)


	@staticmethod
	def accuracy(output, target, topk=(1,)):
		"""Computes the precision@k for the specified values of k"""
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

	@staticmethod
	def get_hms(seconds):
		m, s = divmod(seconds, 60)
		h, m = divmod(m, 60)
		return h, m, s

	#### Fusion
	@staticmethod
	def norm(img):
		min = float(img.min())
		max = float(img.max())
		img = img.clamp_(min=min, max=max)
		img = img.add_(-min).div_(max - min + 1e-5)
		return (img - 0.5) * 2

	@staticmethod
	def combine_npys(npy_folder, out_path):
		print('Combine dataset')
		n = len(os.listdir(npy_folder))
		for k in range(n):
			fpath = os.path.join(npy_folder, "data_{}.npy".format(k))
			if not(os.path.isfile(fpath)):
				continue
			data = np.load(fpath, allow_pickle=True)
			dict_img = data.item()
			img_1 = dict_img['img1']
			img_2 = dict_img['img2']
			img_3 = dict_img['fused']
			t = dict_img['targets']
			if k == 0:
				data_a = img_1
				data_b = img_2
				data_ab = img_3
				targets = t
			else:
				data_a = np.concatenate([data_a, img_1], axis=0)
				data_b = np.concatenate([data_b, img_2], axis=0)
				data_ab = np.concatenate([data_ab, img_3], axis=0)
				targets = np.concatenate([targets, t], axis=0)
		# stack in the 2nd channel
		data = np.stack([data_a, data_b, data_ab], axis=1)
		np.save(out_path, {'data': data, 'targets': targets})
