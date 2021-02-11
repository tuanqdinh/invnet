"""
    @author xxx xxx@cs.xxx.edu
    @date 02/14/2020
"""

from init_invnet import *
import os
import sys
import torch as ch
import numpy as np
import seaborn as sns
from scipy import stats
from tqdm import tqdm, tqdm_notebook
import matplotlib.pyplot as plt
from robustness import model_utils, datasets
import torchvision


# Constants
DATA = 'CIFAR' # Choices: ['CIFAR', 'ImageNet', 'RestrictedImageNet']
dataset_function = getattr(datasets, DATA)
dataset = dataset_function('../data/')
model_kwargs = {
    'arch': 'resnet50',
    'dataset': dataset,
    'resume_path': f'../robust_representations/models/{DATA}.pt'
}
model, _ = model_utils.make_and_restore_model(**model_kwargs)
model = model.to(device)
model.eval()

# Custom loss for inversion
def inversion_loss(model, inp, targ):
    _, rep = model(inp, with_latent=True, fake_relu=True)
    loss = torch.div(torch.norm(rep - targ, dim=1), torch.norm(targ, dim=1))
    return loss, None

# PGD parameters
kwargs = {
    'custom_loss': inversion_loss,
    'constraint':'2',
    'eps':1000,
    'step_size': 1,
    'iterations': 2000, 
    'do_tqdm': True,
    'targeted': True,
    'use_best': False
}


def sample_fused(out_path, dataloader=None, niters=1):
    # Helper.try_make_dir(out_path)
    M = args.batch_size // args.nactors
    N = M * args.nactors
    NOISE_SCALE = 20
    targets, images = [], []
    counter = 0
    nactors = args.nactors
    for epoch in range(niters):
        print('Epoch ', epoch)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print("Epoch {} -- Batch {}".format(epoch, batch_idx))
            inputs, labels = inputs[:N, ...].to(device), labels[:N, ...].to(device)
            (_, z), _ = model(inputs, with_latent=True)
            # Z = [N, 2048]
            z_shape = z.shape
            z = z.unsqueeze(1).contiguous().view([M, args.nactors] + list(z_shape[1:]))
            z_fused = z.mean(dim=1)
            # inverse
            im_n = torch.randn([M] + list(in_shape)) / NOISE_SCALE + 0.5 # Seed for inversion (x_0)
            im_n = im_n.to(device)
            _, x_fused = model(im_n, z_fused, make_adv=True, **kwargs)

            if counter < 3: # save images
                counter += 1
                torchvision.utils.save_image(x_fused.cpu(), '../results/samples/robust-xinv-{}.png'.format(counter), int(args.batch_size**.5), normalize=True)
                torchvision.utils.save_image(inputs.cpu(), '../results/samples/robust-inputs{}.png'.format(counter), int(args.batch_size**.5), normalize=True)

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


## main ###3
fusion_data_dir = os.path.join(args.data_dir, "fusion/{}-{}".format(args.iname, args.nactors))
Helper.try_make_dir(fusion_data_dir)
train_path = os.path.join(fusion_data_dir, 'train')
test_path = os.path.join(fusion_data_dir, 'test')
val_path = os.path.join(fusion_data_dir, 'val')

print("Generating train set")
# sample_fused(train_path, trainloader, niters=args.nactors)
print("Generating test set")
sample_fused(test_path, testloader)
print("Generating validation set")
sample_fused(val_path, valloader)




########################## Store ##################
print('Done')
