import os
from itertools import chain
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from utils.provider import FusionDataset, Provider
import torchvision

def flatten_dict(init_dict):
    res_dict = {}
    if type(init_dict) is not dict:
        return res_dict

    for k, v in init_dict.items():
        if type(v) == dict:
            res_dict.update(flatten_dict(v))
        else:
            res_dict[k] = v
    return res_dict


def setattr_cls_from_kwargs(cls, kwargs):
    kwargs = flatten_dict(kwargs)
    for key in kwargs.keys():
        value = kwargs[key]
        setattr(cls, key, value)


def dict2clsattr(train_configs, model_configs):
    cfgs = {}
    for k, v in chain(train_configs.items(), model_configs.items()):
        cfgs[k] = v

    class cfg_container: pass
    cfg_container.train_configs = train_configs
    cfg_container.model_configs = model_configs
    setattr_cls_from_kwargs(cfg_container, cfgs)
    return cfg_container



def get_loader(fusion_data_dir, name, shuffle=False, batch_size=64, nactors=2): 
    
    max_iters = 60 if nactors == 4 else 30
    
    if name == 'trains':
        data_path = [os.path.join(fusion_data_dir, 'train' + str(count) + '.npy') for count in range(1, max_iters + 1)]
    else:
        data_path = os.path.join(fusion_data_dir, name + '.npy')
    data_set = FusionDataset(data_path)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader

def get_fusion_loaders(fusion_data_dir):
    return get_loader(fusion_data_dir, 'train', True), get_loader(fusion_data_dir, 'val'), get_loader(fusion_data_dir, 'test')

def get_inet_loaders(dataset, data_dir, batch_size=64):
    def get_loader(data, shuffle=False):
        return DataLoader(data, batch_size=batch_size, shuffle=shuffle, drop_last=True)
    # all data
    traindata, testset, in_shape = Provider.load_data(dataset, data_dir)
    nsamples = len(traindata)
    train_size = int(nsamples * 0.8)
    val_size = nsamples - train_size
    trainset, valset = random_split(traindata, [train_size, val_size])
    return get_loader(trainset, True), get_loader(valset), get_loader(testset)

def load_robust_cifar10_model(data_dir):
    DATA = 'cifar10' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
    # dataset_function = getattr(datasets, DATA)
    # dataset = dataset_function(data_dir)
    dataset = CIFAR(data_dir)
    model_kwargs = {
        'arch': 'resnet50',
        'dataset': dataset,
        'resume_path': f'./rb/models/{DATA}.pt'
    }
    model, _ = make_and_restore_model(**model_kwargs)
    return model

def sample_fused_robust_cifar10(args, model, out_path, dataloader=None, niters=1):
    def inversion_loss(model, inp, targ):
        _, rep = model(inp, with_latent=True, fake_relu=True)
        loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
        return loss, None

    # PGD parameters
    kwargs = {
        'custom_loss': inversion_loss,
        'constraint':'unconstrained',
        'eps':1000,
        'step_size': 10,
        'iterations': 5000, 
        'do_tqdm': False,
        'targeted': True,
        'use_best': True
    }
    # Helper.try_make_dir(out_path)
    in_shape = (3, 32, 32)
    M = args.batch_size // args.nactors
    N = M * args.nactors
    NOISE_SCALE = 50
    targets, images = [], []
    counter = 0
    nactors = args.nactors
    for epoch in range(niters):
        print('Epoch ', epoch)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print("Epoch {} -- Batch {}".format(epoch, batch_idx))
            inputs, labels = inputs[:N, ...].to(args.device), labels[:N, ...].to(args.device)
            (_, z), _ = model(inputs, with_latent=True)
            # Z = [N, 2048]
            z_shape = z.shape
            z = z.unsqueeze(1).contiguous().view([M, args.nactors] + list(z_shape[1:]))
            z_fused = z.mean(dim=1)
            # inverse
            im_n = torch.randn([M] + list(in_shape)) / NOISE_SCALE + 0.5 # Seed for inversion (x_0)
            im_n = im_n.to(args.device)
            _, x_fused = model(im_n, z_fused, make_adv=True, **kwargs)

            if counter == 0:
                print('Check pixel')
                print('Inputs: ', inputs.min(), inputs.max(), inputs.mean(), inputs.std())
                print('Targets: ', x_fused.min(), x_fused.max(), x_fused.mean(), x_fused.std())

            if counter < 3: # save images
                counter += 1
                xinv_path = os.path.join(args.inet_sample_dir, 'robust-xinv-{}.png'.format(counter))
                input_path = os.path.join(args.inet_sample_dir, 'robust-inputs{}.png'.format(counter))
                torchvision.utils.save_image(x_fused.cpu(), xinv_path, int(args.batch_size**.5), normalize=True)
                torchvision.utils.save_image(inputs.cpu(), input_path, int(args.batch_size**.5), normalize=True)

            inputs = inputs.view(M, args.nactors, in_shape[0], in_shape[1], in_shape[2])
            labels = labels.view(M, args.nactors)
            data = torch.cat([inputs, x_fused.unsqueeze(1)], dim=1)
            
            #numpy
            data = data.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            images.append(data)
            targets.append(labels)
            #end 
            img_dict = {
                'images': np.concatenate(images, axis=0),
                'labels': np.concatenate(targets, axis=0)
                }
            np.save(out_path, img_dict)
            print('Batch {} - epoch {}'.format(batch_idx, epoch))
            # np.save(out_path + '/{}-{}'.format(epoch, batch_idx), img_dict)
            del inputs, labels, z, z_fused, x_fused, data
        #end
    #end	
    img_dict = {
        'images': np.concatenate(images, axis=0),
        'labels': np.concatenate(targets, axis=0)
        }
    np.save(out_path, img_dict)

    print('Done')


def plot_fused_robust_cifar10(args, model, out_path, dataloader=None):
    def inversion_loss(model, inp, targ):
        _, rep = model(inp, with_latent=True, fake_relu=True)
        loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
        return loss, None

    # PGD parameters
    kwargs = {
        'custom_loss': inversion_loss,
        'constraint':'unconstrained',
        'eps':500,
        'step_size': 10,
        'iterations': 5000, 
        'do_tqdm': False,
        'targeted': True,
        'use_best': True
    }
    # Helper.try_make_dir(out_path)
    in_shape = (3, 32, 32)
    batch_size = args.batch_size
    NOISE_SCALE = 50
    k = args.nactors

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        input_path = os.path.join(args.inet_sample_dir, 'robust-inputs.png')
        torchvision.utils.save_image(inputs.cpu(), input_path, int(batch_size**.5), normalize=True)
        for k in [4]:
            M = batch_size // k
            N = M * k
            inputs, labels = inputs.clone()[:N, ...].to(args.device), labels[:N, ...].to(args.device)
            (_, z), _ = model(inputs, with_latent=True)
            # Z = [N, 2048]
            z_shape = z.shape
            z = z.unsqueeze(1).contiguous().view([M, k] + list(z_shape[1:]))
            z_fused = z.mean(dim=1)
            # inverse
            im_n = torch.randn([M] + list(in_shape)) / NOISE_SCALE + 0.5 # Seed for inversion (x_0)
            im_n = im_n.to(args.device)
            _, x_fused = model(im_n, z_fused, make_adv=True, **kwargs)

            xinv_path = os.path.join(args.inet_sample_dir, 'robust-xinv-{}.png'.format(k))
            torchvision.utils.save_image(x_fused.cpu(), xinv_path, int((batch_size // k)**.5), normalize=True)
        break
    #end	

    print('Done')







def plot_fused_images(save_path, data_loader, fnet, device):
    # images: [N, 4, 3, 32, 32]
    # targets: [N, 3, 32, 32]
    iters = iter(data_loader)
    images, targets = iters.next()
    images, targets = images.to(device), targets.to(device)
    N, A, C, H, W = images.shape
    with torch.no_grad():
        preds = fnet.netG(images.view(N, C*A, H, W))
    data = torch.cat([images, targets.unsqueeze(1), preds.unsqueeze(1)], dim=1)
    data = data.view(images.shape[0] * 6, C, H, W)
    torchvision.utils.save_image(data.cpu(), save_path, nrow=6, normalize=True)
    print('Done Plotting')

