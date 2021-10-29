import os, argparse, json, sys, torch
from train.inv_net.train_inv_net import train_inet
from train.fusion_net.train_fusion_gan_net import train_fusion
from train.fusion_net.ops import evaluate_unet
from networks.fusion_net.pix2pix import Pix2PixModel
from utils.misc import *
from utils.helper import Helper
from utils.tester import iTester
from networks.inv_net.iresnet import conv_iResNet as iResNet

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CodedNet')
## paths
parser.add_argument('--config_file', default='configs', type=str)
parser.add_argument('--data_dir', default='../data')
parser.add_argument('--save_dir', default='../results')
## training
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')

parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', default=0, type=int, help='resume epoch')
parser.add_argument('--nactors', default=2, type=int, help='resume epoch')
parser.add_argument('--nsteps_save', default=1, type=int, help='resume epoch')

parser.add_argument('-densityEstimation', '--densityEstimation', dest='densityEstimation',
                    action='store_true', help='perform density estimation')
parser.add_argument('-sample_fusion', action='store_true')
parser.add_argument('-fusion', action='store_true')
parser.add_argument('-is_train', action='store_false')
parser.add_argument('-test_inversion', action='store_true')
parser.add_argument('-test_fusion', action='store_true')

args = parser.parse_args()


#####  ==== Helpers ===================== #######
def process_args(cfgs):
    # process arguments 
    cfgs.multi_head = True if cfgs.dataset == 'celeba' else False
    cfgs.gpu_ids = [int(e) for e in cfgs.gpu_ids.split(',') if not(e == '')]

    # i-Net 
    mixname = "mixhidden" if cfgs.mixup_hidden else 'vanilla'
    actname = "noact" if cfgs.noActnorm else "act"
    cfgs.inet_name = "{}-{}-{}-{}".format(cfgs.iname, cfgs.dataset, cfgs.nBlocks[0], mixname, actname)
    # f-Net
    cfgs.fnet_name = "{}-{}-{}-{}".format(cfgs.fname, cfgs.netG, cfgs.inet_name, cfgs.nactors)

    # directory
    checkpoint_dir = os.path.join(cfgs.save_dir, 'checkpoints')
    cfgs.inet_save_dir = os.path.join(checkpoint_dir, cfgs.inet_name)
    cfgs.inet_sample_dir = os.path.join(cfgs.save_dir, 'samples/inet')
    Helper.try_make_dir(cfgs.inet_save_dir)
    Helper.try_make_dir(cfgs.inet_sample_dir)

    cfgs.fnet_save_dir = os.path.join(checkpoint_dir, cfgs.fnet_name)
    cfgs.fusion_data_dir = os.path.join(cfgs.data_dir, "fusion/{}".format(cfgs.fnet_name))
    Helper.try_make_dir(cfgs.fnet_save_dir)
    Helper.try_make_dir(cfgs.fusion_data_dir)

    torch.cuda.manual_seed(cfgs.seed)
    cfgs.device = torch.device("cuda:0")

    if cfgs.dataset == 'mnist':
        cfgs.in_shape=(1, 32, 32)
    elif cfgs.dataset == 'cifar10':
        cfgs.in_shape=(3, 32, 32)
    elif cfgs.dataset == 'imagenet':
        cfgs.in_shape=(3, 224, 224)


def load_inet(args, trainloader, device):
    print(args.densityEstimation)
    inet = iResNet(nBlocks=args.nBlocks, nStrides=args.nStrides,
                            nChannels=args.nChannels, nClasses=args.nClasses,
                            init_ds=args.init_ds,
                            inj_pad=args.inj_pad,
                            in_shape=args.in_shape,
                            coeff=args.coeff,
                            numTraceSamples=args.numTraceSamples,
                            numSeriesTerms=args.numSeriesTerms,
                            n_power_iter = args.powerIterSpectralNorm,
                            density_estimation=args.densityEstimation,
                            actnorm=(not args.noActnorm),
                            learn_prior=(not args.fixedPrior),
                            nonlin=args.nonlin, multihead=cfgs.multi_head).to(args.device)
    inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))
    # init actnrom parameters
    init_batch = Helper.get_init_batch(trainloader, args.init_batch)
    print("initializing actnorm parameters...")
    with torch.no_grad():
        inet(init_batch.to(args.device), ignore_logdet=True)
    return inet


def sample_fused(args, model, out_path, dataloader, niters=1):
    # Helper.try_make_dir(out_path)
    in_shape = cfgs.in_shape
    M = args.batch_size // args.nactors
    N = M * args.nactors
    targets, images = [], []
    counter = 0
    nactors = args.nactors
    
    for epoch in range(niters):
        print('Epoch ', epoch)
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            print("Epoch {} -- Batch {}".format(epoch, batch_idx))
            inputs, labels = inputs[:N, ...].cuda(), labels[:N, ...].cuda()
            results = model(inputs)  # Forward Propagation
            zs = results[1]
            # Z = [N, 2048]
            zs_fused = []
            # for i in range(len(zs)):
            z = zs
            z_shape = z.shape
            z = z.unsqueeze(1).contiguous().view([M, nactors] + list(z_shape[1:]))
            z_fused = z.mean(dim=1)
            # zs_fused.append(z_fused)
            # inverse
            x_fused = model.module.inverse(z_fused)
            inputs = inputs.view(M, nactors, in_shape[0], in_shape[1], in_shape[2])
            labels = labels.view(M, nactors)
            data = torch.cat([inputs, x_fused.unsqueeze(1)], dim=1)
            #numpy
            data = data.data.cpu().numpy()
            labels = labels.data.cpu().numpy()
            images.append(data)
            targets.append(labels)
            #end 
        img_dict = {
            'images': np.concatenate(images, axis=0),
            'targets': np.concatenate(targets, axis=0)
            }
        np.save(out_path, img_dict)
        print('Batch {} - epoch {}'.format(batch_idx, epoch))
        # np.save(out_path + '/{}-{}'.format(epoch, batch_idx), img_dict)
        del inputs, labels, zs, zs_fused, x_fused, z, data
        #end
    #end	
    img_dict = {
        'images': np.concatenate(images, axis=0),
        'labels': np.concatenate(targets, axis=0)
        }
    np.save(out_path, img_dict)

    print('Done')



#####  ==== Main ===================== #######

# model configuration
with open(args.config_file) as f:
    model_config = json.load(f)
train_config = vars(args)
cfgs = dict2clsattr(train_config, model_config)
process_args(cfgs)


cfgs.data_dir = '../data/' + str(cfgs.dataset)
loaders = {}
loaders['train'], loaders['val'], loaders['test'] = get_inet_loaders(cfgs.dataset, cfgs.data_dir, cfgs.batch_size)
# invnet 
vTester = iTester(dataloader=loaders['val'], multi_head=cfgs.multi_head)
inet = load_inet(cfgs, loaders['train'], cfgs.device)
inet = inet.to(cfgs.device)
acc1 = -np.inf
if not(args.resume == 0) or cfgs.sample_fusion or cfgs.fusion:
    net_path = os.path.join(cfgs.inet_save_dir, 'best_net.pth')
    if os.path.isfile(net_path):
        print("-- Loading checkpoint '{}'".format(net_path))
        state = torch.load(net_path)
        inet.load_state_dict(state['model'])
        l, acc1, _ = vTester.evaluate(inet)
        print('Evaluating: Loss: %.3f Acc@1: %.3f' % (l, acc1))

if not cfgs.fusion:
    if not cfgs.test_inversion:
        train_inet(cfgs, inet, loaders['train'], vTester, acc1)
else:
    inet.eval()
    if cfgs.sample_fusion:
        for d in ['train', 'val', 'test']:
            print("Generating set " + d)
            fpath = os.path.join(cfgs.fusion_data_dir, d)
            sample_fused(cfgs, inet, fpath, loaders[d], niters=1)

    print('Load model')
    fnet = Pix2PixModel(cfgs)
    fnet.setup(cfgs) 
    # train fusion net
    if cfgs.resume > 0:
        fnet.load_networks()
    
    if cfgs.test_fusion:
        cfgs.is_train = False
        fnet.netG.eval()
        criterion = torch.nn.L1Loss()
        cfgs.fnet_sample_dir = os.path.join(cfgs.save_dir, 'samples/fnet')
        Helper.try_make_dir(cfgs.fnet_sample_dir)
        for s in ['val']:
            data_loader = get_loader(cfgs.fusion_data_dir, s, shuffle=False, batch_size=cfgs.batch_size)
            lossf, corrects = evaluate_unet(data_loader, fnet, inet, criterion, cfgs.device, cfgs.fnet_sample_dir)
            print('{} L1 {:.4f} Acc1: {:.4f}'.format(s, lossf, corrects))
    else:
        print('Load train')
        train_loader = get_loader(cfgs.fusion_data_dir, 'train', shuffle=True, batch_size=cfgs.batch_size, nactors=cfgs.nactors)
        print('Load val')
        val_loader = get_loader(cfgs.fusion_data_dir, 'val', shuffle=False, batch_size=cfgs.batch_size)
        print('Train fusion')
        train_fusion(fnet, inet, train_loader, val_loader, cfgs)
