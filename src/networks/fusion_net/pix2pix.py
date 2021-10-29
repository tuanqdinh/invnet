import torch
import torch.nn.functional as F
import torch.optim as optim

from .base_model import BaseModel
from . import networks

def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))

def loss_hinge_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)

class Pix2PixModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt, inet=None):
        BaseModel.__init__(self, opt)
        self.is_train = opt.is_train
        self.loss_names = ['G_GAN', 'G_L1', 'D', 'G']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.model_names = ['G', 'D'] if self.is_train else ['G']
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nactors=opt.nactors)
        if self.is_train:  
            self.netD = networks.define_D(opt.input_nc * opt.nactors + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0, 0.9))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0, 0.9))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

            self.iteration = 0 
            self.A = opt.nactors
            self.in_shape = opt.in_shape

    def set_dataloader(self, loader):
        self.iterator = loader

    def backward_D(self, inputs, targets, fakes):
        # Fake
        fake_AB = torch.cat((inputs, fakes), 1)  
        pred_fake = self.netD(fake_AB)
        # Real
        real_AB = torch.cat((inputs, targets), 1)
        pred_real = self.netD(real_AB)
        # Loss
        # self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # self.loss_D_real = self.criterionGAN(pred_real, True)
        # self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D = loss_hinge_dis(pred_real, pred_fake)

    def backward_G(self, inputs, targets, fakes):
        # GAN loss
        fake_AB = torch.cat((inputs, fakes), 1)
        pred_fake = self.netD(fake_AB)
        # self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_GAN = loss_hinge_gen(pred_fake)
        # L1 loss
        # self.loss_G_L1 = self.criterionL1(fakes, targets)
        # self.loss_G_Z, self.loss_G_KD = self.rb_loss(self.fake_B, self.real_B, with_kd=True)
        self.loss_G_L1 = torch.cosh(fakes - targets).mean() #log
        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + 100 * self.loss_G_Z + 10 * self.loss_G_KD + self.loss_G_L1 
        self.loss_G = self.loss_G_GAN + self.loss_G_L1

    def train_iter(self):
        self.netG.train()
        self.netD.train()
        C, H, W = self.in_shape
        if self.iteration > 0:
            fakes = self.netG(self.inputs)
            self.backward_G(self.inputs, self.targets, fakes)                 
            self.optimizer_G.zero_grad()        # set G's gradients to zero
            self.loss_G.backward()
            self.optimizer_G.step()             # udpate G's weights  
        for _ in range(5):
            inputs, targets = next(self.iterator)
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = inputs.view(inputs.shape[0], C*self.A, H, W)
            with torch.no_grad():
                fakes = self.netG(inputs)
            self.backward_D(inputs, targets, fakes)       # calculate gradients for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.loss_D.backward()
            self.optimizer_D.step()          # update D's weights

        self.inputs = inputs
        self.targets = targets
        self.fakes = fakes
        self.iteration += 1
