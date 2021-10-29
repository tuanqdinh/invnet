import torch
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F

def loss_hinge_dis(dis_out_real, dis_out_fake):
    return torch.mean(F.relu(1. - dis_out_real)) + torch.mean(F.relu(1. + dis_out_fake))

def loss_hinge_gen(gen_out_fake):
    return -torch.mean(gen_out_fake)

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt, inet=None):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_Z', 'G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_KD']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.is_train = opt.is_train
        self.model_names = ['G', 'D'] if self.is_train else ['G']
        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids, nactors=opt.nactors)
        self.weights = opt.weights
        if self.is_train:  
            # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc * opt.nactors + opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # self.criterionL1 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.9))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        
        self.inet = inet

    def set_input(self, input, target):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_A = input.to(self.device)
        self.real_B = target.to(self.device)
    
    def rb_loss(self, x, y, with_kd=False):
        out_x, z_x, _ = self.inet(x)
        out_y, z_y, _ = self.inet(y)
        if with_kd:
            return self.criterionL1(z_x, z_y), self.kd_loss(out_x, out_y)
        else:
            return self.criterionL1(z_x, z_y)

    def kd_loss(self, x, y):
        T = 6
        return torch.nn.KLDivLoss()(F.log_softmax(x/T, dim=1),
							 F.softmax(y/T, dim=1))

    def z_loss(self, x):
        N, M, H, W = self.real_A.shape
        A = 2; C = 1
        inputs = self.real_A.view(N, A, C, H, W)
        _, z_x, _ = self.inet(inputs[:, 0, ...])
        _, z_y, _ = self.inet(inputs[:, 1, ...])    
        _, z_fused, _ = self.inet(x)
        z_hat = (z_fused - z_y * self.weights[1])/self.weights[0]
        return self.criterionL1(z_hat, z_x)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  

    def backward_D(self):
        # Fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  
        pred_fake = self.netD(fake_AB.detach())
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        # Loss
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        # self.loss_D = loss_hinge_dis(pred_real, pred_fake)
        self.loss_D.backward()

    def backward_G(self):
        # GAN loss
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # self.loss_G_GAN = loss_hinge_gen(pred_fake)

        # L1 loss
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_Z, self.loss_G_KD = 0, 0
        # self.loss_G_Z, self.loss_G_KD  = self.rb_loss(self.fake_B, self.real_B, with_kd=True)
        # if self.weights:
            # self.loss_G_Z = self.z_loss(self.fake_B)
        # else:
        self.loss_G_Z, self.loss_G_KD  = self.rb_loss(self.fake_B, self.real_B, with_kd=True)    
        # self.loss_logcosh = torch.log(torch.cosh(self.fake_B - self.real_B)).mean()

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + 100 * self.loss_G_Z + 10 * self.loss_G_KD + self.loss_G_L1 
        # self.loss_G = self.loss_G_Z
        # self.loss_G = self.loss_G_GAN + 10 * self.loss_G_L1 
        self.loss_G.backward()

    def optimize_parameters(self):
        self.netG.train()
        self.netD.train()
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
