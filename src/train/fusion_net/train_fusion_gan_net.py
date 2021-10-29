import torch, time, shutil, os
from train.fusion_net.ops import evaluate_unet
from utils.helper import Helper
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

class InfDataLoader():
    def __init__(self, data_loader, **kwargs):
        self.dataloader = data_loader
        def inf_dataloader():
            while True:
                for data in self.dataloader:
                    image, label = data
                    yield image, label
        self.inf_dataloader = inf_dataloader()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.inf_dataloader)

    def __del__(self):
        del self.dataloader

def log_norm(logf, inputs, targets, pred):
    Helper.log(logf,'Input: {:.4f} {:.4f} {:.4f} {:.4f}'.format(inputs.max(), inputs.min(), inputs.mean(), inputs.std()))
    Helper.log(logf,'Target:{:.4f} {:.4f} {:.4f} {:.4f}'.format(targets.max(), targets.min(), targets.mean(), targets.std()))
    Helper.log(logf,'Predict: {:.4f} {:.4f} {:.4f} {:.4f}'.format(pred.max(), pred.min(), pred.mean(), pred.std()))

def log_loss(logf, e, batch_idx, total_steps, losses):
    # Helper.log(logf,"===> Epoch[{}]({}/{}): D-real: {:.4f} D-fake: {:.4f} G-GAN: {:.4f} G-L1 {:.4f} G-Z {:.4f} G-KD {:.4f}".format(epoch, batch_idx, total_steps, losses['D_real'], losses['D_fake'], losses['G_GAN'], losses['G_L1'], losses['G_Z'], losses['G_KD']))
    Helper.log(logf, "Epoch[{}]({}/{}) G: {:.4f} D: {:.4f}  L1: {:.4f}".format(e, batch_idx, total_steps, losses['G'], losses['D'], losses['G_L1']))

def train_fusion(fnet, inet, train_loader, val_loader, cfgs):
    logf = open('../logs/train_{}.out'.format(cfgs.fnet_name), 'w')
    log_dir = '../logs/' + str(cfgs.fnet_name)
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)

    elapsed_time = 0
    criterion = torch.nn.L1Loss()
    total_steps = len(iter(train_loader))
    lossf, corrects = evaluate_unet(val_loader, fnet, inet, criterion, cfgs.device)
    Helper.log(logf,'Initial Evaluation: L1 {:.4f} ACCURACY: {:.4f}'.format(lossf, corrects))
    max_corrects = corrects
    iterator = InfDataLoader(train_loader)
    fnet.set_dataloader(iterator)
    for e in range(cfgs.epochs):
        start_time = time.time()
        for i in range(total_steps):
            fnet.train_iter()
            if i > 0 and i % cfgs.log_steps == 0: 
                losses = fnet.get_current_losses()
                log_loss(logf, e, i, total_steps, losses)

        lossf, corrects = evaluate_unet(val_loader, fnet, inet, criterion, cfgs.device)
        Helper.log(logf,'== Val denst {:.4f}  acc: {:.4f}'.format(lossf, corrects))
        if max_corrects < corrects:
            max_corrects = corrects
            Helper.log(logf, 'saving the best fnet (epoch %d, total_iters %d)' % (e, total_steps))
            fnet.save_networks(e)
        
        epoch_time = time.time() - start_time
        elapsed_time += epoch_time
        Helper.log(logf,'| Elapsed time : %d:%02d:%02d' % (Helper.get_hms(elapsed_time)))
        # writer.add_scalar('l1/train', losses['G_L1'], e)
        writer.add_scalar('l1/val', lossf, e)
        writer.add_scalar('acc/val', corrects, e)
        fnet.update_learning_rate()
        torch.cuda.empty_cache()
    Helper.log(logf,'Done')
    writer.close()


def train_fnet(fnet, inet, train_loader, val_loader, cfgs):
    criterion = torch.nn.L1Loss()
    optim_fnet = optim.Adam(fnet.parameters(), lr=cfgs.lr, betas=(cfgs.beta1, 0.999))
    # fnet_scheduler = Helper.get_scheduler(optim_fnet, cfgs)
    total_steps = len(iter(train_loader))
    max_corrects  = 0
    for e in range(cfgs.epochs):
        epoch = e + 1
        for batch_idx, (images, targets) in enumerate(train_loader):
            inputs, targets = images.to(cfgs.device), targets.to(cfgs.device)
            N, A, C, H, W = inputs.shape
            inputs = inputs.view(N, C*A, H, W)

            optim_fnet.zero_grad()
            preds = fnet(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            
            # log out
            if batch_idx % cfgs.log_steps == 0:
                Helper.log(logf,"===> Epoch[{}]({}/{}): L1-pixel: {:.4f}".format(epoch, batch_idx, total_steps, loss.item()))

        lossf, corrects = evaluate_unet(val_loader, fnet, inet, criterion, cfgs.device)
        Helper.log(logf,'Evaluation: L1 {:.4f} Acc1: {:.4f}'.format(lossf, corrects))
        if max_corrects < corrects:
            max_corrects = corrects
            Helper.log(logf,'saving the best fnet (epoch %d, total_iters %d)' % (epoch, total_steps))
            # fnet_path = os.path.join(checkpoint_dir, '{}/{}_{}_e{}.pth'.format(root, root, cfgs.model_name, cfgs.resume_g + epoch))
            # torch.save(fnet.state_dict(), fnet_path)
    Helper.log(logf,'Done')
