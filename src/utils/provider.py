
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class FusionDataset(Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, data_path, transform=None, use_tanh=False):
		"""
		Args:
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		dict_data = np.load(data_path, allow_pickle=True)
		dict_data = dict_data.item()
		images = dict_data['images'].astype('float32')
		self.data = torch.Tensor(images[:, :-1,  ...])
		if use_tanh:
			self.targets = torch.tanh(torch.Tensor(images[:, -1, ...]))
		else:
			self.targets = torch.Tensor(images[:, -1, ...])
		self.transform = transform
		# stats = {
		#     'target': {},
		#     'input':{}
		# }
		# stats['target']['mean'] = np.mean(data[:, -1, ...], axis=(0, 2,3))
		# stats['target']['std'] = np.std(data[:, -1, ...], axis=(0, 2,3))
		# stats['target']['min'] = np.min(data[:, -1, ...], axis=(0, 2,3))
		# stats['target']['max'] = np.max(data[:, -1, ...], axis=(0, 2,3))
		#
		# stats['input']['mean'] = np.mean(data[:, 0, ...], axis=(0, 2,3))
		# stats['input']['std'] = np.std(data[:, 0, ...], axis=(0, 2,3))
		# stats['input']['min'] = np.min(data[:, 0, ...], axis=(0, 2,3))
		# stats['input']['max'] = np.max(data[:, 0, ...], axis=(0, 2,3))
		# # STORE stats
		# print(stats)
		# np.save(stat_path, stats)
		# ############ Standarlize ##############
		# for i in range(args.input_nc):
		#     for k in range(args.nactors):
		#         data[:, k, i, ...] = (data[:, k, i, ...] - stats['input']['mean'][i]) / stats['input']['std'][i]
		#     data[:, -1, i, ...] = (data[:, -1, i, ...] - stats['target']['mean'][i]) / stats['target']['std'][i]

	def __len__(self):
		return len(self.targets)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		data = self.data[idx, ...]
		targets = self.targets[idx, ...]
		samples = (data, targets)
		if self.transform:
			samples = self.transform(samples)

		return samples


class Provider:
	@staticmethod
	def load_data(dataset, data_dir):
		########## Data ####################
		if dataset == 'cifar10':
			# 'cifar10':
			in_shape = (3, 32, 32)
			mean = {
				'cifar10': (0.4914, 0.4822, 0.4465),
				'cifar100': (0.5071, 0.4867, 0.4408),
			}

			std = {
				'cifar10': (0.2023, 0.1994, 0.2010),
				# 'cifar10': (0.229, 0.224, 0.225),
				'cifar100': (0.2675, 0.2565, 0.2761),
			}

			# Only for cifar-10
			classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

			train_chain = [transforms.Pad(4, padding_mode="symmetric"),
						   transforms.RandomCrop(32),
						   transforms.RandomHorizontalFlip(),
						   transforms.ToTensor()]
			test_chain = [transforms.ToTensor()]
			clf_chain = [transforms.Normalize(mean[dataset], std[dataset])]
			transform_train = transforms.Compose(train_chain + clf_chain)
			transform_test = transforms.Compose(test_chain + clf_chain)
			trainset = torchvision.datasets.CIFAR10(
				root=data_dir, train=True, download=True, transform=transform_train)
			testset = torchvision.datasets.CIFAR10(
				root=data_dir, train=False, download=True, transform=transform_test)
		else:
			dens_est_chain = [
				lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
				lambda x: x / 256.,
				lambda x: x - 0.5
			]
			# mnist_transforms = [transforms.Pad(2, 0), transforms.ToTensor(), lambda x: x.repeat((3, 1, 1))]
			mnist_transforms = [transforms.Pad(2, 0), transforms.ToTensor()]
			transform_train_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
			transform_test_mnist = transforms.Compose(mnist_transforms + dens_est_chain)
			if dataset == 'fashion':
				trainset = torchvision.datasets.FashionMNIST(
					root=data_dir, train=True, download=True, transform=transform_train_mnist)
				testset = torchvision.datasets.FashionMNIST(
					root=data_dir, train=False, download=True, transform=transform_test_mnist)
			else:
				trainset = torchvision.datasets.MNIST(
					root=data_dir, train=True, download=True, transform=transform_train_mnist)
				testset = torchvision.datasets.MNIST(
					root=data_dir, train=False, download=True, transform=transform_test_mnist)
			in_shape = (1, 32, 32)

		return trainset, testset, in_shape
