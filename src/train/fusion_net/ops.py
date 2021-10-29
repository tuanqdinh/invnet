import torch, os, time
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from networks.inv_net.iresnet import conv_iResNet as iResNet
from utils.provider import Provider
from utils.helper import Helper
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
# from utils.tester import atanh

bce_loss = nn.BCELoss()
softmax = nn.Softmax(dim=1)
criterionKLD = nn.KLDivLoss()

def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.

def load_networks(net, resume, net_save_dir):
    if resume == -1:
        save_filename = 'best_net.pth'
    else:
        save_filename = '%s_net.pth' % (resume)
    net_path = os.path.join(net_save_dir, save_filename)
    init_epoch = 0
    if os.path.isfile(net_path):
        print("-- Loading checkpoint '{}'".format(net_path))
        state = torch.load(net_path)
        net.load_state_dict(state['model'])
        return True, net
    else:
        print("--- No checkpoint found at '{}'".format(net_path))
        return False, None

def load_iresnet(args, trainloader, device):
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
                            nonlin=args.nonlin).to(device)
                            
    inet = torch.nn.DataParallel(inet, range(torch.cuda.device_count()))

    # init actnrom parameters
    init_batch = Helper.get_init_batch(trainloader, args.init_batch)
    print("initializing actnorm parameters...")
    with torch.no_grad():
        inet(init_batch.to(args.device), ignore_logdet=True)

    if args.fused or not(args.resume == 0):
        return load_networks(inet, -1, args.inet_save_dir)
        # return load_imagenet_resume(inet, args.inet_save_dir)

    return True, inet


def normalize(tensor, value_range=None, scale_each=False):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if value_range is not None:
        assert isinstance(value_range, tuple), \
            "value_range has to be a tuple (min, max) if specified. min and max are numbers"

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)
    return tensor 
    
def plot_fused_images(save_path, images, targets, preds, x_hat, x_hat_2):
    data = torch.cat([images, targets.unsqueeze(1), preds.unsqueeze(1), x_hat.unsqueeze(1), x_hat_2.unsqueeze(1)], dim=1)
    data = data.view(images.shape[0] * 6, 3, 224, 224)
    torchvision.utils.save_image(data.cpu(), save_path, nrow=6, normalize=False)
    print('Done Plotting')
    

def evaluate_time(data_loader, fnet, inet, criterion, device, fnet_sample_dir=None):
    lossf = []
    corrects = 0
    total = 0
    inet.eval()
    fnet.netG.eval()
    f_times = []
    g_times = []
    for batch_idx, (images, targets) in enumerate(data_loader):
        inputs = images.to(device)
        N, A, C, H, W = inputs.shape
        data = inputs.view(N * A, C, H, W)
        start_time = time.time()
        z = inet.module.embed(data)
        end_f_time = time.time()
        inet.module.classifier(z)
        end_g_time = time.time()

        f_time = end_f_time - start_time
        g_time = end_g_time - end_f_time
        # Classification evaluation
        f_times.append(f_time)
        g_times.append(g_time)
    
    f_times = np.asarray(f_times)
    g_times = np.asarray(g_times)
    print("F-time: {:.4f} {:.4f} {:.4f}".format(np.median(f_times), np.mean(f_times), np.std(f_times)))
    print("G-time: {:.4f} {:.4f} {:.4f}".format(np.median(g_times), np.mean(g_times), np.std(g_times)))




def evaluate_unet(data_loader, fnet, inet, criterion, device, fnet_sample_dir=None):
    lossf = []
    corrects = 0
    total = 0
    inet.eval()
    fnet.netG.eval()
    for batch_idx, (images, targets) in enumerate(data_loader):
        inputs = images.to(device)
        targets = targets.to(device)
        N, A, C, H, W = inputs.shape
        fake = fnet.netG(inputs.view(N, C*A, H, W))
        loss_l1 = criterion(fake, targets)
        lossf.append(loss_l1.data.cpu().numpy())

        # Classification evaluation
        _, z_c, _ = inet(inputs.view(N * A, C, H, W))
        z_c = z_c.view(N, A, z_c.shape[1], z_c.shape[2], z_c.shape[3])
        z_0 = z_c[:, 0, ...]
        z_c = z_c[:, 1:, ...].sum(dim=1)
        # img_fused = atanh(fake)
        img_fused = fake # no tanh
        _, z_fused, _ = inet(img_fused)
        z_hat = A * z_fused - z_c

        out_hat = inet.module.classifier(z_hat)
        out = inet.module.classifier(z_0)
        _, y = torch.max(out.data, 1)
        _, y_hat = torch.max(out_hat.data, 1)
        correct = y_hat.eq(y.data).sum().cpu().numpy()
        corrects += correct / N

        # density 
        # x_hat = inet.module.inverse(z_hat)
        # _, _, _, = inet(x_hat)
        # logpx = logpz + trace
        # loss_denest = bits_per_dim(logpx, inputs).mean()
        # lossf.append(loss_denest.item())

        total += 1
        

    return np.mean(lossf), corrects/total

def evaluate_unet_robustness(data_loader, fnet, inet, criterion, device):
    lossf = []
    corrects = 0
    total = 0
    # inet.eval()
    # fnet.netG.eval()
    with torch.no_grad():
        for _, (images, targets) in enumerate(data_loader):
            inputs = images.to(device)
            targets = targets.to(device)
            N, A, C, H, W = inputs.shape
            # inputs = inputs[:, torch.randperm(A), ...]
            fake = fnet.netG(inputs.view(N, C*A, H, W))
            # import ipdb; ipdb.set_trace()
            # added for robustness
            loss_l1 = criterion(fake, targets)
            lossf.append(loss_l1.data.cpu().numpy())

            # Classification evaluation
            # with torch.no_grad():
            #     # z_c: 2048
            #     (_, z_c), _ = inet(inputs.view(N * A, C, H, W), with_latent=True)
            (_, z_c), _ = inet(inputs.view(N * A, C, H, W), with_latent=True)

            z_c = z_c.view(N, A, 2048)
            z_0 = z_c[:, 0, ...]
            z_c = z_c[:, 1:, ...].sum(dim=1)
            # img_fused = atanh(fake)
            img_fused = fake # no tanh
            (_, z_fused), _ = inet(img_fused, with_latent=True)
            z_hat = A * z_fused - z_c

            out_hat = inet.classify(z_hat)

            out = inet.classify(z_0)

            _, y = torch.max(out.data, 1)
            _, y_hat = torch.max(out_hat.data, 1)
            correct = y_hat.eq(y.data).sum().cpu().numpy()
            corrects += correct / N
            total += 1

    return np.mean(lossf), corrects/total

def evaluate_robustness_inversion(data_loader, inet, criterion, device):
    lossf = []
    corrects = 0
    total = 0
    inet.eval()
    for _, (images, targets) in enumerate(data_loader):
        inputs = images.to(device)
        targets = targets.to(device)
        N, A, C, H, W = inputs.shape
        # Classification evaluation
        with torch.no_grad():
            # z_c: 2048
            (_, z_c), _ = inet(inputs.view(N * A, C, H, W), with_latent=True)
        z_c = z_c.view(N, A, 2048)
        z_0 = z_c[:, 0, ...]
        z_c = z_c[:, 1:, ...].sum(dim=1)

        img_fused = targets
        (_, z_fused), _ = inet(img_fused, with_latent=True)
        z_hat = A * z_fused - z_c

        out_hat = inet.classify(z_hat)

        out = inet.classify(z_0)

        _, y = torch.max(out.data, 1)
        _, y_hat = torch.max(out_hat.data, 1)
        correct = y_hat.eq(y.data).sum().cpu().numpy()
        corrects += correct / N
        total += 1

    return -1, corrects/total


def evaluate_dnet(data_loader, fnet, inet, device, nactors, nClasses):
    corrects = []
    A = nactors
    fnet.eval()
    for _, (inputs, labels) in enumerate(data_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        N, C, H, W = inputs.shape
        M = N // A
        N = M * A
        inputs = inputs[:N, ...]
        labels = labels[:N, ...]

        logits, _, _ = inet(inputs)
        logits = logits.view(M, A, nClasses)
        logit_missed = logits[:, 0, :]
        logit_rest = logits[:, 1:, :].sum(dim=1)
        # fusion inputs
        x_g = inputs.view(M, A, C, H, W).view(M, A*C, H, W)
        logits_mean_g, _, _ = fnet(x_g)
        logits_missed_g = logits_mean_g * A - logit_rest

        _, y = torch.max(logit_missed.data, 1)
        _, y_hat = torch.max(logits_missed_g.data, 1)
        # dangerous => 0
        correct = y_hat.eq(y.data).sum().cpu().numpy() / M
        corrects.append(correct)

    return np.mean(corrects)

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, temperature):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    T = temperature # small T for small network student
    beta = (1. - alpha) * T * T # alpha for student: small alpha is better
    teacher_loss = criterionKLD(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1))
    # student_loss = F.cross_entropy(outputs, labels)
    student_loss = bce_loss(softmax(outputs), labels)
    KD_loss =  beta * teacher_loss + alpha * student_loss

    return KD_loss
