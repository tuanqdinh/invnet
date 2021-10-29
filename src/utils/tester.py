import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import os, time, sys
import numpy as np

from utils.helper import Helper, AverageMeter


###===================== Coordinator =====================####
class Coordinator():
    def __init__(self, out_type=None):
        super(Coordinator, self).__init__()

        if out_type == 'polar':
            self.forward = self.cart2polar
            self.backward = self.polar2cart
        elif out_type == 'sphere':
            self.forward = self.cart2sphere
            self.backward = self.sphere2cart
        elif out_type == 'angle':
            self.forward = self.cart2angle
            self.backward = self.angle2cart
        else:
            self.forward = lambda x: x
            self.backward = lambda x: x

    def cart2polar(self, p):
        # p = [N, D]
        # angles: reverse [x_d, ... , x_1]
        r = torch.norm(p, dim=1)
        s = torch.sqrt(torch.cumsum(p**2, dim=1))
        angles = torch.acos(p/s)
        ## [N, D-1]
        angles = angles[:, 1:]
        return r, angles

    def polar2cart(self, r, angles):
        # r= [N], angles=[N, D-1]
        p = torch.zeros(angles.shape[0], angles.shape[1] + 1).cuda()
        prev = torch.ones(angles.shape[0]).cuda()
        for i in range(1, p.shape[1]):
            k = p.shape[1] - i
            p[:, k] = prev * torch.cos(angles[:, k-1])
            prev = prev * torch.sin(angles[:, k-1])
        p[:, 0] = prev
        p = p * r.unsqueeze(1).expand(p.shape)
        return p

    def convert_polar_to_cart(self, z):
        p0 = [cart2polar(z[i, :]) for i in range(M*A)]
        p1 = torch.stack(p0, dim=0)
        r = torch.norm(z, dim=1)
        p = p1.view(M, A, C*H*W-1).mean(dim=1)
        r = r.view(M, A).mean(dim=1)
        z_fused = torch.stack([polar2cart(r[i], p[i, :]) for i in range(M)], dim=0)

        # hypersphere
        # [NA, d+1]
        p = cart2sphere(z)
        p = p.view(M, A, C_z*H_z*W_z + 1)
        # [N, d+1]
        p_fused = p.mean(dim = 1)
        z_fused = sphere2cart(p_fused)

    def cart2sphere(self, p):
        # x = [N, D]
        r = torch.norm(p, dim=1)**2
        y = (r - 1)/(r + 1)
        r = r.unsqueeze(1).expand(p.shape)
        x = 2*p/(r + 1)
        s = torch.cat([x, y.unsqueeze(1)], dim=1)
        return s

    def sphere2cart(self, s):
        x = s[:, :-1]
        y = s[:, -1:].expand(x.shape)
        p = x/(1 - y)
        return p

    def cart2angle(self, p):
        # input: p=[N, D], output: c = [N, D]
        r = torch.norm(p, dim=1)
        s = r.unsqueeze(1).expand(p.shape)
        angles = torch.acos(p/s)
        return r, angles

    def angle2cart(self, r, angles):
        # input: r=[N], angles=[N, D], output: p = [N, D]
        r = r.unsqueeze(1).expand(angles.shape)
        p = torch.cos(angles) * r
        return p

###===================== TestHelper =====================####

class TestHelper:
    @staticmethod
    def atanh(x, eps=1e-4):
        return 0.5*torch.log((1+x + eps)/(1-x + eps))

    @staticmethod
    def scale(imgs, stats=None):
        if stats == None:
            return torch.tanh(imgs)
        else:
            s = stats['input']
            for k in range(nchannels):
                imgs[:, k, ...] = (imgs[:, k, ...] - s['mean'][k])/s['std'][k]
            return imgs

    @staticmethod
    def rescale(imgs, stats=None, use_tanh=False):
        if use_tanh:
            return TestHelper.atanh(imgs)
        elif stats:
            s = stats['target']
            for k in range(nchannels):
                imgs[:, k, ...] = imgs[:, k, ...] * s['std'][k] + s['mean'][k]
        return imgs



###===================== Processor Tester =====================####
class Processor():
    def __init__(self, inet, nactors, fuse_type, weights=None):
        super(Processor, self).__init__()
        self.inet = inet
        self.A = nactors
        if fuse_type == 'polar':
            self.coord = Coordinator('polar')
            self.fuse = self.fuse_polar
            self.reconstruct = self.reconstruct_polar
        else:
            self.weights = np.asarray(weights)

    def process_data(self, inputs, labels, reshape=False):
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        N = (inputs.shape[0] // self.A) * self.A
        self.M = N // self.A
        inputs = inputs[:N, ...]
        labels = labels[:N, ...]

        return inputs, labels

    def process_latent(self, inputs, labels, missed=True):
        _, z, _ = self.inet(inputs)
        self.z_shape = z.shape
        z = z.unsqueeze(1).contiguous().view([self.M, self.A] + list(self.z_shape[1:]))

        if missed:
            z_missed = z[:, 0, ...]
            z_rest = z[:, 1:, ...]
            targets = labels.unsqueeze(1).contiguous().view(self.M, self.A)
            targets_missed = targets[:, 0]
            return z, z_missed, z_rest, targets_missed
        else:
            return z

    def fuse(self, z):
        y = z[:, 0, ...] * self.weights[0] 
        for i in range(1, self.A):
            y += z[:, i, ...] * self.weights[i] 
        return y

    def fuse_polar(self, z):
        self.D = self.z_shape[1] * self.z_shape[2] * self.z_shape[3]
        z_input = z.reshape([self.M*self.A] + list(self.z_shape[1:])).reshape(self.M*self.A, self.D)
        r, angles = self.coord.forward(z_input)
        r = r.view(self.M, self.A)
        angles = angles.view(self.M, self.A, self.D-1)
        r_mean = torch.exp(torch.log(r).mean(dim=1))
        angle_mean = angles.mean(dim=1)
        z_fused = self.coord.backward(r_mean, angle_mean)
        z_fused = z_fused.view([self.M] + list(self.z_shape[1:]))
        return z_fused

    def reconstruct_polar(self, z_fused_hat, z_rest):
        z_rest = z_rest.reshape([self.M *(self.A-1)] + list(self.z_shape[1:])).reshape(sefl.M * (self.A-1), self.D)
        r_rest, angle_rest = self.coord.forward(z_rest)
        r_rest = r_rest.view(self.M, self.A-1)
        angle_rest = angle_rest.view(self.M, self.A-1, self.D-1)

        r_mean_hat, angle_mean_hat = self.coord.forward(z_fused_hat.view(self.M, self.D))
        angle_missed = self.A*angle_mean_hat - angle_rest.sum(dim=1)
        r_missed = self.A * r_mean_hat - r_rest.sum(dim=1)
        z_missed_hat = self.coord.backward(r_missed, angle_missed)
        z_missed_hat = z_missed_hat.view(M, Cz, Hz, Wz)
        return z_missed_hat

    def batch_score(self, z_missed, z_missed_g, targets):
        out_missed_g = self.inet.module.classifier(z_missed_g)
        out_missed = self.inet.module.classifier(z_missed)
        # evaluation 1
        scores = [0, 0]
        _, y = torch.max(out_missed.data, 1)
        _, y_g = torch.max(out_missed_g.data, 1)
        scores[0] = y.eq(targets.data).sum().cpu().item()
        scores[1] = y_g.eq(targets.data).sum().cpu().item()

        return np.asarray(scores)

###===================== IResNet Tester =====================####
class iTester():
    def __init__(self, inet=None, iname=None, in_shapes=None, dataloader=None, nactors=None, fuse_type=None, weights=None, multi_head=False):
        super(iTester, self).__init__()
        self.inet = inet
        self.iname = iname
        self.multi_head = multi_head
        self.dataloader = dataloader
        self.nactors = nactors
        self.in_shapes = in_shapes
        self.criterionCE = nn.CrossEntropyLoss()
        self.processor = Processor(self.inet, nactors, fuse_type, weights)
        if not(inet == None):
            self.inet.eval()

    def set_inet(self, model):
        self.inet = model
        self.inet.eval()


    def validate(self, model):
        def accuracy(output, target, topk=(1,)):
            """Computes the precision@k for the specified values of k"""
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


        criterion = self.criterionCE
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()
        classes = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
        end = time.time()
        for i, (input, target) in enumerate(self.dataloader):
            target = target.cuda()
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

            # compute output
            output, out_bij, _ = model(input_var)
            # map to 
            output = output[:, classes]
        
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.data, input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(self.dataloader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
            .format(top1=top1, top5=top5))

        return top1.avg

    def evaluate(self, net=None):
        if net == None:
            net = self.inet
        net.eval()
        def bits_per_dim(logpx, inputs):
            return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.
        # loss
        losses = AverageMeter()
        top1 = AverageMeter()
        top2 = AverageMeter()
        for _, (inputs, targets) in enumerate(self.dataloader):
            # print(batch_idx)
            targets = Variable(targets).cuda()
            inputs = Variable(inputs).cuda()
            results = net.module.evaluate(inputs) # logits, z, logpx, trace
            if self.multi_head:
                loss = self.criterionCE(results[0], targets[:, 0])
                prec1, _ = Helper.accuracy(results[0], targets[:, 0], topk=(1,2))
                prec2, _ = Helper.accuracy(results[1], targets[:, 1], topk=(1,2))
            else:
                loss = self.criterionCE(results[0], targets)
                precs = Helper.accuracy(results[0], targets, topk=(1,))
                prec1 = precs[0]

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

        return losses.avg, top1.avg, None

    def eval_inv(self):
        print('Evaluate Invertibility on ', self.iname)
        # loss
        ticks = time.time()
        corrects = np.zeros(2) # correct-x, correct-x-hat, match
        total = 0
        for batch_idx, (inputs, labels) in enumerate(self.dataloader):
            inputs, labels = self.processor.process_data(inputs, labels)
            z, z_missed, z_rest, target_missed = self.processor.process_latent(inputs, labels, missed=True)
            z_fused = self.processor.fuse(z)
            x_fused = self.inet.module.inverse(z_fused)
            _, z_fused_hat, _ = self.inet(x_fused)
            z_missed_hat = self.processor.reconstruct(z_fused_hat, z_rest)

            corr = self.processor.batch_score(z_missed, z_missed_hat, target_missed)
            corrects += corr

            total += z.shape[0]
            
            del z, z_missed, z_rest, target_missed, x_fused, z_fused_hat, z_missed_hat

        corrects = 100 * corrects / total
        print('\t Correctly classified: X {:.4f} X_hat {:.4f}'.format(corrects[0], corrects[1]))
        return corrects[0]

    def plot_fused(self, sample_dir, dataloader=None):
        if dataloader == None:
            dataloader = self.dataloader

        (inputs, labels) = iter(self.dataloader).next()
        inputs, labels = self.processor.process_data(inputs, labels)
        z = self.processor.process_latent(inputs, labels, missed=False)
        z_fused = self.processor.fuse(z)
        x_fused = self.inet.module.inverse(z_fused)

        Helper.save_images(x_fused, sample_dir, self.iname, 'fused', 0)
        print("pixel range")
        print("x: {:.4f} {:.4f} {:.4f}".format(inputs.min(), inputs.median(), inputs.max()))
        print("x-fused: {:.4f} {:.4f} {:.4f}".format(x_fused.min(), x_fused.median(), x_fused.max()))
        print("z: {:.4f} {:.4f} {:.4f}".format(z.min(), z.median(), z.max()))
        print("z-fused: {:.4f} {:.4f} {:.4f}".format(z_fused.min(), z_fused.median(), z_fused.max()))

    def sample_fused(self, out_path, dataloader=None, niters=1):
        if dataloader==None:
            dataloader = self.dataloader
        images = None
        targets = None
        first = True
        for epoch in range(niters):
            print('Epoch ', epoch)
            for batch_idx, (inputs, labels) in enumerate(dataloader):
                inputs, labels = self.processor.process_data(inputs, labels)
                z = self.processor.process_latent(inputs, labels, missed=False)
                z_fused = self.processor.fuse(z)
                x_fused = self.inet.module.inverse(z_fused)
                M = inputs.shape[0] // self.nactors
                inputs = inputs.view(M, self.nactors, inputs.shape[1], inputs.shape[2], inputs.shape[3])
                labels = labels.view(M, self.nactors)
                data = torch.cat([inputs, x_fused.unsqueeze(1)], dim=1)
                data = data.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
                if first:
                    first = False
                    images = data
                    targets = labels
                else:
                    images = np.concatenate((images, data), axis=0)
                    targets = np.concatenate((targets, labels), axis=0)

        img_dict = {
            'images': images,
            'labels': targets
            }
        np.save(out_path, img_dict)

        print('Done')



    ###===================== Fusion Tester =====================####
    def evaluate_fusion(self, fnet):
        fnet.eval()
        # loss
        corrects = np.zeros(2) # correct-x, correct-x-hat, match
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.dataloader):
            
            inputs, labels = self.processor.process_data(inputs, labels)
            z, z_missed, z_rest, target_missed = self.processor.process_latent(inputs, labels, missed=True)
            M = inputs.shape[0] // self.nactors
            _, C, D, W = inputs.shape
            inputs = inputs.view(M, self.nactors, C, D, W)
            x_g = inputs.view(M, self.nactors*C, D, W)
            x_fused_hat = fnet(x_g)

            _, z_fused_hat, _ = self.inet(x_fused_hat)
            z_missed_hat = self.processor.reconstruct(z_fused_hat, z_rest)

            corr = self.processor.batch_score(z_missed, z_missed_hat, target_missed)
            corrects += corr

            total += z.shape[0]
            del z, z_missed, z_rest, target_missed, z_fused_hat, z_missed_hat

        corrects = 100 * corrects / total
        print('\t Correctly classified: fine-class {:.4f} super-class {:.4f}'.format(corrects[0], corrects[1]))
        return corrects 

    def evaluate_fgan(self, fnet):
        # loss
        fi_time = []
        fusion_time = []
        g_time = []
        for batch_idx, (inputs, labels) in enumerate(self.dataloader):
            inputs, labels = self.processor.process_data(inputs, labels)
            z, z_missed, z_rest, target_missed = self.processor.process_latent(inputs, labels, missed=True)

            #### difference
            M = inputs.shape[0] // self.nactors
            _, C, D, W = inputs.shape
            inputs = inputs.view(M, self.nactors, C, D, W)
            x_g = inputs.view(M, self.nactors*C, D, W)
            fusion_start = time.time()
            x_fused_hat = fnet(x_g)
            fusion_time.append(time.time() - fusion_start)
            ############

            fi_start = time.time()
            _, z_fused_hat, _ = self.inet(x_fused_hat)
            fi_time.append(time.time() - fi_start)
            z_missed_hat = self.processor.reconstruct(z_fused_hat, z_rest)

            g_start = time.time()
            corr = self.processor.batch_score(z_missed, z_missed_hat, target_missed)
            g_time.append(time.time() - g_start)

        g_time = np.asarray(g_time)*1000
        fi_time = np.asarray(fi_time)*1000
        fusion_time = np.asarray(fusion_time)*1000
        print('Score: ', corr)
        print(self.iname, self.nactors)
        print('Gi: {:.3} {:.3}'.format(np.median(g_time), g_time.std()))
        print('Fi: {:.3} {:.3}'.format(np.median(fi_time), fi_time.std()))
        print('Fu: {:.3} {:.3}'.format(np.median(fusion_time), fusion_time.std()))

    def evaluate_fgan2(self, fnet):
        # loss
        corrects = np.zeros(2) # correct-x, correct-x-hat, match
        total = 0
        fusion_time = []
        g_time = []
        for batch_idx, (inputs, labels) in enumerate(self.dataloader):
            inputs, labels = self.processor.process_data(inputs, labels)
            z, z_missed, z_rest, target_missed = self.processor.process_latent(inputs, labels, missed=True)

            #### difference
            M = inputs.shape[0] // self.nactors
            _, C, D, W = inputs.shape
            inputs = inputs.view(M, self.nactors, C, D, W)
            x_g = inputs.view(M, self.nactors*C, D, W)
            fusion_start = time.time()
            x_fused_hat = fnet(x_g)
            fusion_time.append(time.time() - fusion_start)
            ############

            fi_start = time.time()
            _, z_fused_hat, _ = self.inet(x_fused_hat)
            fi_time.append(time.time() - fi_start)
            z_missed_hat = self.processor.reconstruct(z_fused_hat, z_rest)

            g_start = time.time()
            corr = self.processor.batch_score(z_missed, z_missed_hat, target_missed)
            g_time.append(time.time() - g_start)
            corrects += corr

            total += z.shape[0]
            del z, z_missed, z_rest, target_missed, z_fused_hat, z_missed_hat

        corrects = 100 * corrects / total
        print('\t Correctly classified: fine-class {:.4f} super-class {:.4f}'.format(corrects[0], corrects[1]))
        return corrects

    def plot_reconstruction(self, fnet, fname, sample_dir):
        (inputs, labels) = iter(self.dataloader).next()
        inputs, labels = self.processor.process_data(inputs, labels)
        z, z_missed, z_rest, _ = self.processor.process_latent(inputs, labels, missed=True)

        #### difference
        M = inputs.shape[0] // self.nactors
        N, C, D, W = inputs.shape
        input_g = inputs.view(M, self.nactors, C, D, W)
        input_g = input_g.view(M, self.nactors*C, D, W)
        x_fused_g = fnet(input_g)
        ############

        _, z_fused_g, _ = self.inet(x_fused_g)
        z_missed_g = self.processor.reconstruct(z_fused_g, z_rest)

        # inverse
        self.inet.eval()
        z_fused = self.processor.fuse(z)
        x_fused = self.inet.module.inverse(z_fused)
        x_missed_g = self.inet.module.inverse(z_missed_g)
        x_missed = self.inet.module.inverse(z_missed)

        # visualize
        Helper.save_images(x_fused_g, sample_dir, fname, 'x_fused_g', 0)
        Helper.save_images(x_fused, sample_dir, fname, 'x_fused', 0)
        Helper.save_images(x_missed_g, sample_dir, fname, 'x_missed_g', 0)
        Helper.save_images(x_missed, sample_dir, fname, 'x_missed', 0)
