
import torch, os
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import numpy as np

class FusionDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data_path, transform=None, use_tanh=False):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if type(data_path) == list:
            images = []
            for fpath in data_path:
                dict_data = np.load(fpath, allow_pickle=True)
                dict_data = dict_data.item()
                imgs = dict_data['images'].astype('float32')
                images.append(imgs)
            images = np.concatenate(images, axis=0)
        else:
            dict_data = np.load(data_path, allow_pickle=True)
            dict_data = dict_data.item()
            images = dict_data['images'].astype('float32')
        data = images[:, :-1,  ...]
        self.targets = torch.Tensor(images[:, -1, ...])
        self.data = torch.Tensor(data) if transform == None else transform(data)
        if use_tanh:
            self.targets = torch.tanh(self.targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        data = self.data[idx, ...]
        targets = self.targets[idx, ...]
        return (data, targets)


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

            train_chain = [transforms.Pad(4, padding_mode="symmetric"),
                           transforms.RandomCrop(32),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor()]
            test_chain = [transforms.ToTensor()]

            # clf_chain = [transforms.Normalize(mean[dataset], std[dataset])]
            # transform_train = transforms.Compose(train_chain + clf_chain)
            # transform_test = transforms.Compose(test_chain + clf_chain)

            dens_est_chain = [
                lambda x: (255. * x) + torch.zeros_like(x).uniform_(0., 1.),
                lambda x: x / 256.,
                lambda x: x - 0.5
            ]
            transform_train = transforms.Compose(train_chain + dens_est_chain)
            transform_test = transforms.Compose(test_chain + dens_est_chain)
            # clf_chain = [transforms.Normalize(mean[dataset], std[dataset])]
            # transform_train = transforms.Compose(train_chain + clf_chain)
            # transform_test = transforms.Compose(test_chain + clf_chain)

            trainset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform_train)
            testset = torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform_test)
        elif dataset == 'imagenet':
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            traindir = os.path.join(data_dir, 'train')
            valdir = os.path.join(data_dir, 'val')
            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.87, contrast=0.5,
                                        saturation=0.5, hue=0.2),
                    transforms.ToTensor(),
                    normalize,
            ]))
            testset = datasets.ImageFolder(
                valdir, 
                transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
            ]))
            in_shape = (3, 224, 224)
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
