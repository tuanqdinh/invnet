"""
    @author xxx xxx@cs.wisc.edu
    @date 02/14/2020
"""
import os, sys, time, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from utils.helper import Helper, AverageMeter
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import warnings
warnings.filterwarnings("ignore")


def bits_per_dim(logpx, inputs):
    return -logpx / float(np.log(2.) * np.prod(inputs.shape[1:])) + 8.

def train_inet(cfgs, inet, trainloader, vTester=None, best_accuracy=-np.inf):
    logf = open('../logs/train_{}.out'.format(cfgs.inet_name), 'w')
    log_dir = '../logs/multitask_' + str(cfgs.inet_name)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    # model
    optimizer = optim.Adam(inet.parameters(), lr=cfgs.lr)
    criterionCE = nn.CrossEntropyLoss()
    bce_loss = nn.BCELoss()
    softmax = nn.Softmax(dim=1)

    nsteps = len(iter(trainloader))
    Helper.log(logf,'Model: '+ cfgs.iname)
    Helper.log(logf, '|  Train: mixup: {} mixup_hidden {} alpha {} epochs {}'.format(cfgs.mixup, cfgs.mixup_hidden, cfgs.mixup_alpha, cfgs.epochs))
    Helper.log(logf, '|  Initial Learning Rate: ' + str(cfgs.lr))
    elapsed_time = 0
    best_epoch = 0
    for epoch in range(cfgs.epochs):
        start_time = time.time()
        inet.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        top2 = AverageMeter()

        # update lr for this epoch (for classification only)
        lr = Helper.learning_rate(cfgs.lr, epoch, factor=30)
        Helper.update_lr(optimizer, lr)
        Helper.log(logf, '|Learning Rate: ' + str(lr))
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            cur_iter = (epoch - 1) * len(trainloader) + batch_idx
            # if first epoch use warmup
            if epoch - 1 <= cfgs.warmup_epochs:
                this_lr = cfgs.lr * float(cur_iter) / (cfgs.warmup_epochs * len(trainloader))
                # Helper.log(logf, '|Learning Rate - warmup: ' + str(this_lr))
                Helper.update_lr(optimizer, this_lr)

            inputs = Variable(inputs).to(cfgs.device)
            targets = Variable(targets).to(cfgs.device)
            if cfgs.multi_head:
                results = inet(inputs)
                loss1 = criterionCE(results[0], targets[:, 0])
                loss2 = criterionCE(results[1], targets[:, 1])
                prec1, _ = Helper.accuracy(results[0], targets[:, 0], topk=(1,2))
                prec2, _ = Helper.accuracy(results[1], targets[:, 1], topk=(1,2))
                loss = loss1 + loss2
            else:
                # mixup
                if cfgs.mixup_hidden:
                    logits, _, reweighted_targets = inet(inputs, targets=targets, mixup_hidden=cfgs.mixup_hidden, mixup=cfgs.mixup, mixup_alpha=cfgs.mixup_alpha)
                    # classification
                    _, labels = torch.max(reweighted_targets.data, 1)
                    precs = Helper.accuracy(logits, labels, topk=(1,))
                    prec1 = precs[0]
                    loss = bce_loss(softmax(logits), reweighted_targets)
                else:
                    logits, _, _ = inet(inputs)
                    prec1, prec2 = Helper.accuracy(logits, targets, topk=(1,2))
                    loss = criterionCE(logits, targets)
                
            # measure accuracy and record loss
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % cfgs.log_steps == 0:
                Helper.log(logf, '| Epoch [%3d/%3d] Iter[%3d/%3d]\tLoss: %.4f Acc@1: %.3f'
                            % (epoch, cfgs.epochs, batch_idx+1, nsteps, losses.avg, top1.avg))

        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        Helper.log(logf, '| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))

        # evaluate
        l, acc1, nll = vTester.evaluate(inet)
        Helper.log(logf, 'Evaluating: Loss: %.3f Acc@1: %.3f' % (l, acc1))
        if acc1 > best_accuracy:
            best_accuracy = acc1
            is_best = True
            best_epoch = epoch
            Helper.log(logf, 'The best net')
        else: 
            is_best = False
        Helper.save_networks(inet, cfgs.inet_save_dir, epoch, cfgs, loss, prec1, l, acc1, is_best)
        writer.add_scalar('loss/train', loss.item(), epoch)
        writer.add_scalar('loss/val', l, epoch)
        writer.add_scalar('acc_clf/train', top1.avg, epoch)
        writer.add_scalar('acc_clf/val', acc1, epoch)

        torch.cuda.empty_cache()
                

    ########################## Store ##################
    # end of training
    Helper.log(logf, 'Best validation: {:.4f} @ epoch {}'.format(best_accuracy, best_epoch))
    writer.close()
