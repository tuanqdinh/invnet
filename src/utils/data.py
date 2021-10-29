import os
import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
from torchvision import datasets, transforms


def load_data(dataset, labels, train=True, normal=False):

    batch_size = 64

    if dataset.lower() == "mnist":

        root = 'MNIST_cDCGAN_results/'
        if not os.path.isdir(root):
            os.mkdir(root)

        img_size = 32

        if normal:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])

        trainset = datasets.MNIST('MNIST_cDCGAN_results/data', train=train, download=True, transform=transform)

        data = []
        for label in labels:
            indices = np.argwhere(np.array(trainset.targets) == label)
            indices = indices.reshape(len(indices))
            data.append(torch.utils.data.Subset(trainset, indices))
    
        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data), batch_size=batch_size)

    elif dataset.lower() == "cifar":

        root = '../data'
        if not os.path.isdir(root):
            os.mkdir(root)

        img_size = 32

        transform = transforms.Compose([
				transforms.ToTensor(),
                ])

        trainset = datasets.CIFAR10('../data', train=train, download=True, transform=transform)

        data = []
        for label in labels:
            indices = np.argwhere(np.array(trainset.targets) == label)
            indices = indices.reshape(len(indices))
            data.append(torch.utils.data.Subset(trainset, indices))
        
        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data), batch_size=batch_size)

    dataset = []
    labelset = []
    for data, label in train_loader:
        dataset.append(data)
        labelset.append(label)
        
    dataset = torch.cat(dataset, 0)
    labelset = torch.cat(labelset, 0)

    return dataset, labelset

def load_random_data(data_name, train=True):
    if train:
        transform_3D = transforms.Compose([
								transforms.RandomCrop(32, padding=4), 
							    transforms.RandomHorizontalFlip(),
								transforms.ToTensor(),
								#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
								])
    else:
        transform_3D = transforms.Compose([
								transforms.ToTensor(),
								#transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
                                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
								])

    transform_1D = transforms.Compose([
                                transforms.Resize((32, 32)), 
                                transforms.ToTensor(), 
                                transforms.Normalize([0.5], [0.5])
                                ])

    if not os.path.isdir('data'):
        os.mkdir('data')

    data_path = os.path.join('data', data_name)

    if not os.path.isdir(data_path):
        os.mkdir(data_path)

    if data_name == 'mnist':
        return datasets.MNIST(data_path, train=train, download=True, transform=transform_1D)
    elif data_name == 'cifar10':
        return datasets.CIFAR10('../data', train=train, download=True, transform=transform_3D)
    else:
        raise NotImplementedError





def old_load_data(dataset, labels, train=True, normal=False, noise=False, clip=False):

    batch_size = 64

    if dataset.lower() == "mnist":

        root = 'MNIST_cDCGAN_results/'
        if not os.path.isdir(root):
            os.mkdir(root)

        img_size = 32

        if normal:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])

        trainset = datasets.MNIST('MNIST_cDCGAN_results/data', train=train, download=True, transform=transform)

        data = []
        for label in labels:
            indices = np.argwhere(np.array(trainset.targets) == label)
            indices = indices.reshape(len(indices))
            data.append(torch.utils.data.Subset(trainset, indices))
    
        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data), batch_size=batch_size)

    elif dataset.lower() == "cifar":

        def _noise_adder(img):
            return torch.empty_like(img, dtype=img.dtype).uniform_(0.0, 1/255.0) + img

        root = '../'
        if not os.path.isdir(root):
            os.mkdir(root)

        img_size = 32

        if normal:
            if clip:
                transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4), 
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
                ])
            else:
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), 
                ])

        elif noise:
            transform = transforms.Compose([
				transforms.ToTensor(),
                _noise_adder,
            ])
        else:
            transform = transforms.Compose([
				transforms.ToTensor(),
            ])

        print(transform)
        trainset = datasets.CIFAR10('../data', train=train, download=True, transform=transform)

        data = []
        for label in labels:
            indices = np.argwhere(np.array(trainset.targets) == label)
            indices = indices.reshape(len(indices))
            data.append(torch.utils.data.Subset(trainset, indices))
        
        train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(data), batch_size=batch_size)

    dataset = []
    labelset = []
    for data, label in train_loader:
        dataset.append(data)
        labelset.append(label)
        
    dataset = torch.cat(dataset, 0)
    labelset = torch.cat(labelset, 0)

    return dataset, labelset






# generate extend dataset for MNIST
# include digit 7&9
def generate_extend_dataset(extend=False):

    root = 'MNIST_cDCGAN_results/'
    if not os.path.isdir(root):
        os.mkdir(root)

    batch_size = 60000

    img_size = 32
    transform = transforms.Compose([
        transforms.Scale(img_size),
        transforms.ToTensor(),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('MNIST_cDCGAN_results/data', train=True, download=True, transform=transform),
        batch_size=batch_size)

    for data, label in train_loader:
        merged_train_data = torch.cat([data[label == 7], data[label == 9]], 0)
        merged_train_label = torch.cat([label[label == 7], label[label == 9]], 0)

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('MNIST_cDCGAN_results/data', train=False, download=True, transform=transform),
        batch_size=batch_size)

    for data, label in train_loader:
        merged_test_data = torch.cat([data[label == 7], data[label == 9]], 0)
        merged_test_label = torch.cat([label[label == 7], label[label == 9]], 0)

    if extend == True:

        extend_train_data = torch.cat([merged_train_data,
                                       merged_train_data * 0.25,
                                       merged_train_data * 0.5,
                                       merged_train_data * 0.75], 0)

        extend_train_label = torch.cat([merged_train_label,
                                        merged_train_label,
                                        merged_train_label,
                                        merged_train_label], 0)

        extend_test_data = torch.cat([merged_test_data,
                                      merged_test_data * 0.25,
                                      merged_test_data * 0.5,
                                      merged_test_data * 0.75], 0)

        extend_test_label = torch.cat([merged_test_label,
                                       merged_test_label,
                                       merged_test_label,
                                       merged_test_label], 0)

        np.save('MNIST_cDCGAN_results/data/extend_train_data.npy', extend_train_data.numpy())
        np.save('MNIST_cDCGAN_results/data/extend_train_label.npy', extend_train_label.numpy())
        np.save('MNIST_cDCGAN_results/data/extend_test_data.npy', extend_test_data.numpy())
        np.save('MNIST_cDCGAN_results/data/extend_test_label.npy', extend_test_label.numpy())

        return extend_train_data, extend_train_label, extend_test_data, extend_test_label

    else:

        np.save('MNIST_cDCGAN_results/data/train_data.npy', merged_train_data.numpy())
        np.save('MNIST_cDCGAN_results/data/train_label.npy', merged_train_label.numpy())
        np.save('MNIST_cDCGAN_results/data/test_data.npy', merged_test_data.numpy())
        np.save('MNIST_cDCGAN_results/data/test_label.npy', merged_test_label.numpy())

        return merged_train_data, merged_train_label, merged_test_data, merged_test_label


if __name__ == '__main__':
    generate_extend_dataset(True)
