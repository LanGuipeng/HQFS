#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: LanGuipeng

PyTorch = 1.10
"""

import warnings
warnings.filterwarnings("ignore")

from argparse import ArgumentParser
import pprint
from torch.utils.data import DataLoader
from utils.images_dataset import ImagesDataset
import torchvision.transforms as transforms
import random
import torch
import torch.nn as nn
import math
from model.encoder import GradualStyleEncoder
from model.decoder import Generator
from model.mapper import mapping_network
from utils.ranger import Ranger
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from utils import function
import matplotlib.pyplot as plt
# from model import VGG, expression_consistent
from torchvision.utils import make_grid
from skimage.transform import resize
from PIL import Image
import numpy as np

from loss_function import id_loss
from loss_function.lpips.lpips import LPIPS
# from loss_function.get_landmark import LandmarkLoss

random.seed(0)
torch.manual_seed(0)

class TrainOptions:
    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        # Basic parameters 
        self.parser.add_argument('--exp_dir', default='results/', help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='ffhq_encode', type=str,
                                 help='Type of dataset/experiment to run')
        self.parser.add_argument('--output_size', default=512, type=int, help='Output size of generator')
        self.parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--lpips_type', default='alex', type=str, help='LPIPS backbone')
        self.parser.add_argument('--workers', default=0, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=0, type=int,
                                 help='Number of test/inference dataloader workers')
        self.parser.add_argument('--stylegan_size', default=512, type=int,
                                 help='size of pretrained StyleGAN Generator')
        # loss weight
        self.parser.add_argument('--expression_lambda', default=0.8, type=float, help='Landmark loss multiplier factor')
        self.parser.add_argument('--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=10.0, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--l2_lambda', default=0.8, type=float, help='L2 loss multiplier factor')
        # Pretrained model path
        self.parser.add_argument('--stylegan_weights', default='pretained_model/stylegan2-ffhq-config-f.pt', type=str,
                                 help='Path to StyleGAN model weights')
        # The parameters of training process 
        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to tensorboard')
        self.parser.add_argument('--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=10000, type=int, help='Model checkpoint interval')
        self.parser.add_argument('--train_decoder', default=False, type=bool, help='Whether to train the decoder model')
        # Discriminator flags
        self.parser.add_argument('--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
        self.parser.add_argument('--w_discriminator_lr', default=2e-5, type=float, help='Dw learning rate')
        self.parser.add_argument("--r1", type=float, default=10, help="weight of the r1 regularization")
        self.parser.add_argument("--d_reg_every", type=int, default=16,
                                 help="interval for applying r1 regularization")
        self.parser.add_argument('--use_w_pool', action='store_true',
                                 help='Whether to store a latnet codes pool for the discriminator\'s training')
        self.parser.add_argument("--w_pool_size", type=int, default=50,
                                 help="W\'s pool size, depends on --use_w_pool")
    def parse(self):
        opts = self.parser.parse_args()
        return opts

opts = TrainOptions().parse()
opts_dict = vars(opts)
pprint.pprint(opts_dict)
global_step = 0
device = 'cuda:0'
opts.device = device
'''
if os.path.exists(opts.exp_dir):
    raise Exception('Oops... {} already exists'.format(opts.exp_dir))
os.makedirs(opts.exp_dir)
'''
# Loading dataset
# transforms
transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_test': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }

def configure_datasets():
    train_dataset = ImagesDataset(source_root='/usr/local/tarfile/lan/datasets/A',
                                  target_root='/usr/local/tarfile/lan/datasets/B',
                                  source_transform=transforms_dict['transform_source'],
                                  target_transform=transforms_dict['transform_gt_train'])
    test_dataset = ImagesDataset(source_root='/usr/local/tarfile/lan/datasets/test/A',
                                 target_root='/usr/local/tarfile/lan/datasets/test/B',
                                 source_transform=transforms_dict['transform_source'],
                                 target_transform=transforms_dict['transform_test'])
    print("Number of training samples: {}".format(len(train_dataset)))
    print("Number of test samples: {}".format(len(test_dataset)))
    return train_dataset, test_dataset
          
train_dataset, test_dataset = configure_datasets()
train_dataloader = DataLoader(train_dataset,
                                   batch_size=opts.batch_size,
                                   shuffle=True,
                                   num_workers=int(opts.workers),
                                   drop_last=True)
test_dataloader = DataLoader(test_dataset,
                                  batch_size=opts.test_batch_size,
                                  shuffle=False,
                                  num_workers=int(opts.test_workers),
                                  drop_last=True)
# --------------------------------------Networks------------------------------------- #
def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class Encoder(nn.Module):
    def __init__(self, opts):
        super(Encoder, self).__init__()
        self.opts = opts
        self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
        # Define architecture
        self.encoder = GradualStyleEncoder()
        # Load weights if needed
        self.mask = nn.Sequential(nn.Conv1d(self.opts.n_styles,self.opts.n_styles,1), nn.InstanceNorm1d(self.opts.n_styles), nn.Sigmoid())
        # self.load_weights()
        '''
    def load_weights(self):
        print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load('pretained_model/model_ir_se50.pth')
        self.encoder.load_state_dict(encoder_ckpt, strict=False)
        '''
    def forward(self, x, y):
        x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor = self.encoder(x, y)

        return x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor

class Decoder(nn.Module):
    def __init__(self, opts):
        super(Decoder, self).__init__()
        self.opts = opts
        # Define architecture
        self.decoder = Generator(opts.stylegan_size, 512, 8, channel_multiplier=2)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        # self.load_weights()
    '''
    def load_weights(self):
        print('Loading decoder weights from pretrained!')
        ckpt = torch.load(self.opts.stylegan_weights)
        self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
        self.__load_latent_avg(ckpt, repeat=2*int(math.log(opts.stylegan_size, 2))-2)
    '''
    def forward(self, codes, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)
        
        return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

encoder = Encoder(opts).to(opts.device)
decoder = Decoder(opts).to(opts.device)
mapping_net_high = mapping_network(3).cuda()
mapping_net_medium = mapping_network(4).cuda()
mapping_net_low = mapping_network(9).cuda()


from model.encoder import Backbone
facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6)
facenet.load_state_dict(torch.load('pretained_model/model_ir_se50.pth'))
facenet.cuda().eval()
print('identity extractor is loading!')
face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
face_pool_256 = torch.nn.AdaptiveAvgPool2d((256, 256))
# -----------------------------------------base---------------------------------------- #
# optimizer
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def configure_optimizers():
    params = list(mapping_net_high.parameters())
    params = list(mapping_net_medium.parameters())
    params += list(mapping_net_low.parameters())
    #params += list(encoder.parameters())
    if opts.train_decoder:
        params += list(decoder.parameters())
    else:
        requires_grad(decoder, False)
    if opts.optim_name == 'adam':
        optimizer = torch.optim.Adam(params, lr=opts.learning_rate)
    else:
        optimizer = Ranger(params, lr=opts.learning_rate)
    return optimizer

optimizer = configure_optimizers()

# Initialize logger
log_dir = os.path.join(opts.exp_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)
logger = SummaryWriter(log_dir=log_dir)
# Initialize checkpoint dir
checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)
best_val_loss = None
if opts.save_interval is None:
    opts.save_interval = opts.max_steps
# --------------------------------------loss function------------------------------------- #
from model.face_model import ReconNetWrapper

class ShapeAwareIdentityExtractor(nn.Module):
    def __init__(self, f_3d_checkpoint_path):
        super(ShapeAwareIdentityExtractor, self).__init__()
        self.f_3d = ReconNetWrapper(net_recon='resnet50', use_last_fc=False)
        self.f_3d.load_state_dict(torch.load(f_3d_checkpoint_path, map_location='cpu')['net_recon'])
        self.f_3d.eval()

    def forward(self, i_source, i_target):
        c_s = self.f_3d(i_source)
        c_t = self.f_3d(i_target)

        return c_s[:, 80:], c_t[:, 80:]
extractor = ShapeAwareIdentityExtractor('pretained_model/epoch_20.pth')

if opts.lpips_lambda > 0:
    lpips_loss = LPIPS(net_type=opts.lpips_type).to(device).eval()
if opts.id_lambda > 0:
    id_loss = id_loss.IDLoss().to(device).eval()
mse_loss = nn.MSELoss().to(device).eval()
# ---------------------------Auxiliary function of training process------------------------- #
# landmarkloss = LandmarkLoss().cuda()

from loss_function.perceptualloss import PerceptualLoss
perceptual_loss = PerceptualLoss(layer_weights={'conv1_2': 0.1,'conv2_2': 0.1,'conv3_4': 1,'conv4_4': 1,'conv5_4': 1}).cuda()

def calc_loss(x, y, y_hat, global_step):
    loss_dict = {}
    loss = 0.0
    id_logs = None

    if opts.id_lambda > 0:
        loss_id, sim_improvement, id_logs = id_loss(face_pool_256(y_hat), face_pool_256(y), face_pool_256(x)) # note: identity_x 2 y
        loss_dict['loss_id'] = float(loss_id)
        loss_dict['id_improve'] = float(sim_improvement)
        loss = loss_id * opts.id_lambda
    if opts.l2_lambda > 0:
        loss_l2 = F.mse_loss(y_hat, x)
        loss_dict['loss_l2'] = float(loss_l2)
        loss += loss_l2 * opts.l2_lambda
    if opts.lpips_lambda > 0:
        loss_lpips = lpips_loss(y_hat, x)
        loss_dict['loss_lpips'] = float(loss_lpips)
        loss += loss_lpips * opts.lpips_lambda
    '''
    if global_step % 4 == 0:
        loss_self_recon = F.mse_loss(x, x_self_recon)
        loss_dict['loss_self_recon'] = float(loss_self_recon)
        loss += loss_self_recon*20
    
    loss_landmark = landmarkloss(y_hat, x)
    loss_dict['loss_landmark'] = float(loss_landmark)
    loss += loss_landmark*20
    '''
    source_expression, target_expression = extractor(y_hat.cpu(), x.cpu())
    loss_3d_recon = F.mse_loss(source_expression, target_expression)
    loss_dict['loss_3d_recon'] = float(loss_3d_recon)
    loss += loss_3d_recon*0.1
    '''
    loss_perceptual = perceptual_loss(y_hat, x)
    loss_dict['loss_perceptual'] = float(loss_perceptual[0])
    loss += loss_perceptual[0]*0.1
    '''
    loss_dict['loss'] = float(loss)
    return loss, loss_dict, id_logs

def parse_and_log_images(id_logs, x, y, y_hat, title, subscript=None, display_count=1):
    im_data = []
    for i in range(display_count):
        cur_im_data = {
            'input_face': function.log_input_image(x[i], opts),
            'target_face': function.tensor2im(y[i]),
            'output_face': function.tensor2im(y_hat[i]),
        }
        if id_logs is not None:
            for key in id_logs[i]:
                cur_im_data[key] = id_logs[i][key]
        im_data.append(cur_im_data)
    log_images(title, im_data=im_data, subscript=subscript)

def log_images(name, im_data, subscript=None, log_latest=False):
    fig = function.vis_faces(im_data)
    step = global_step
    if log_latest:
        step = 0
    if subscript:
        path = os.path.join(logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
    else:
        path = os.path.join(logger.log_dir, name, '{:04d}.jpg'.format(step))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    
def log_metrics(metrics_dict, prefix):
    for key, value in metrics_dict.items():
        logger.add_scalar(f'{prefix}/{key}', value, global_step)
    

def print_metrics(metrics_dict, prefix):
    print(f'Metrics for {prefix}, step {global_step}')
    for key, value in metrics_dict.items():
        print(f'\t{key} = ', value)


def validate():
    encoder.cuda().eval()
    decoder.cuda().eval()
    mapping_net_low.cuda().eval()
    mapping_net_medium.cuda().eval()
    mapping_net_high.cuda().eval()
    ##mapping_net_medium_level.cuda().eval()
    agg_loss_dict = []
    for batch_idx, batch in enumerate(test_dataloader):
        x, y = batch

        with torch.no_grad():
            x = x.cuda().float()
            y = y.cuda().float()
            
            # identity y
            y_resize = face_pool_256(y)
            y_face = y_resize[:, :, 35:223, 32:220]  # Crop interesting     256: [:, :, 35:223, 32:220]
            y_face = face_pool(y_face)
            identity_y = facenet(y_face)
            
            x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor = encoder(face_pool_256(x), face_pool_256(y))

            x_high_code_mapping_edit = x_high_code_tensor + mapping_net_high(x_high_code_tensor, identity_y)
            x_medium_code_mapping_edit = x_medium_code_tensor + mapping_net_medium(x_medium_code_tensor, identity_y)
            x_low_code_mapping_edit = x_low_code_tensor + mapping_net_low(x_low_code_tensor, identity_y)

            x_high_code_edit = torch.cat((x_high_code_mapping_edit, x_medium_code_mapping_edit, x_low_code_mapping_edit), dim=1)
            y_hat = decoder(x_high_code_edit, return_latents=True)
            loss, cur_loss_dict, id_logs = calc_loss(x, y, y_hat, global_step)
        agg_loss_dict.append(cur_loss_dict)

        # Logging related
        parse_and_log_images(id_logs, x, y, y_hat,
                                title='images/test/faces',
                                subscript='{:04d}'.format(batch_idx))

        # For first step just do sanity test on small amount of data
        if global_step == 0 and batch_idx >= 4:
            mapping_net_high.train()
            mapping_net_medium.train()
            mapping_net_low.train()
            return None  # Do not log, inaccurate in first batch

    loss_dict = function.aggregate_loss_dict(agg_loss_dict)
    log_metrics(loss_dict, prefix='test')
    print_metrics(loss_dict, prefix='test')

    mapping_net_high.train().cuda()
    mapping_net_medium.train().cuda()
    mapping_net_low.train().cuda()
    return loss_dict

def checkpoint_me(loss_dict, is_best):
    save_name = 'best_model.pt' if is_best else f'iteration_{global_step}.pt'
    save_dict = __get_save_dict()
    checkpoint_path = os.path.join(checkpoint_dir, save_name)
    torch.save(save_dict, checkpoint_path)
    with open(os.path.join(checkpoint_dir, 'timestamp.txt'), 'a') as f:
        if is_best:
            f.write(f'**Best**: Step - {global_step}, Loss - {best_val_loss} \n{loss_dict}\n')
            f.write(f'Step - {global_step}, \n{loss_dict}\n')

def __get_save_dict():
    save_dict = {
        'encoder_state_dict': encoder.state_dict(),
        # 'identity_trans_dict':identity_trans.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'opts': vars(opts)
    }
    # save the latent avg in state_dict for inference if truncation of w was used during training
    if opts.start_from_latent_avg:
        save_dict['latent_avg'] = encoder.latent_avg, decoder.latent_avg
    return save_dict
# --------------------------------------Discriminator-------------------------------------- #

class LatentCodesDiscriminator(nn.Module):
    def __init__(self, style_dim, n_mlp):
        super().__init__()
        self.style_dim = style_dim
        layers = []
        for i in range(n_mlp-1):
            layers.append(
                nn.Linear(style_dim, style_dim)
            )
            layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(512, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, w):
        return self.mlp(w)

import random
class LatentCodesPool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_ws = 0
            self.ws = []

    def query(self, ws):
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return ws
        return_ws = []
        for w in ws:  # ws.shape: (batch, 512) or (batch, n_latent, 512)
            # w = torch.unsqueeze(image.data, 0)
            if w.ndim == 2:
                i = random.randint(0, len(w) - 1)  # apply a random latent index as a candidate
                w = w[i]
            self.handle_w(w, return_ws)
        return_ws = torch.stack(return_ws, 0)   # collect all the images and return
        return return_ws

    def handle_w(self, w, return_ws):
        if self.num_ws < self.pool_size:  # if the buffer is not full; keep inserting current codes to the buffer
            self.num_ws = self.num_ws + 1
            self.ws.append(w)
            return_ws.append(w)
        else:
            p = random.uniform(0, 1)
            if p > 0.5:  # by 50% chance, the buffer will return a previously stored latent code, and insert the current code into the buffer
                random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                tmp = self.ws[random_id].clone()
                self.ws[random_id] = w
                return_ws.append(tmp)
            else:  # by another 50% chance, the buffer will return the current image
                return_ws.append(w)

discriminator = LatentCodesDiscriminator(512, 4).cuda()
discriminator_optimizer = torch.optim.Adam(list(discriminator.parameters()), lr=opts.w_discriminator_lr)
real_w_pool = LatentCodesPool(opts.w_pool_size)
fake_w_pool = LatentCodesPool(opts.w_pool_size)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def discriminator_loss(real_pred, fake_pred, loss_dict):
    real_loss = F.softplus(-real_pred).mean()
    fake_loss = F.softplus(fake_pred).mean()

    loss_dict['d_real_loss'] = float(real_loss)
    loss_dict['d_fake_loss'] = float(fake_loss)

    return real_loss + fake_loss

from torch import autograd
def discriminator_r1_loss(real_pred, real_w):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_w, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

# -----------------------------------------Training---------------------------------------- #

# training from staring
encoder.cuda()
decoder.cuda()
mapping_net_high.cuda()
mapping_net_medium.cuda()
mapping_net_low.cuda()
encoder_path = 'pretained_model/model_based_on_pSp_encoder300000.pth'
decoder_path = 'pretained_model/model_based_on_pSp_decoder300000.pth'
encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
encoder.eval()
decoder.eval()
mapping_net_high.train()
mapping_net_medium.train()
mapping_net_low.train()
'''
encoder.cuda()
decoder.cuda()
mapping_net_high.cuda()
mapping_net_medium.cuda()
mapping_net_low.cuda()

encoder_path = 'pretained_model/model_based_on_pSp_encoder300000.pth'
decoder_path = 'pretained_model/model_based_on_pSp_decoder300000.pth'
mapping_net_high_path = 'output/model_based_on_pSp_mapping_net_high_level80000.pth'
mapping_net_medium_path = 'output/model_based_on_pSp_mapping_net_medium_level80000.pth'
mapping_net_low_path = 'output/model_based_on_pSp_mapping_net_low_level80000.pth'

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
mapping_net_high.load_state_dict(torch.load(mapping_net_high_path))
mapping_net_medium.load_state_dict(torch.load(mapping_net_medium_path))
mapping_net_low.load_state_dict(torch.load(mapping_net_low_path))
'''
encoder.eval()
decoder.eval()
mapping_net_high.train()
mapping_net_medium.train()
mapping_net_low.train()
print('model has already been loaded!')
while global_step < opts.max_steps:
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        x, y = batch
        x, y = x.cuda().float(), y.cuda().float()
        
        # identity x
        x_resize = face_pool_256(x)
        x_face = x_resize[:, :, 35:223, 32:220] # Crop interesting region
        x_face = face_pool(x_face)
        identity_x = facenet(x_face)
        # identity y
        y_resize = face_pool_256(y)
        y_face = y_resize[:, :, 35:223, 32:220]  # Crop interesting region
        y_face = face_pool(y_face)
        identity_y = facenet(y_face)
        
        x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor = encoder(face_pool_256(x), face_pool_256(y))
        x_high_code_mapping_edit = x_high_code_tensor + mapping_net_high(x_high_code_tensor, identity_y)
        x_medium_code_mapping_edit = x_medium_code_tensor + mapping_net_medium(x_medium_code_tensor, identity_y)
        x_low_code_mapping_edit = x_low_code_tensor + mapping_net_low(x_low_code_tensor, identity_y)

        x_high_code_edit = torch.cat((x_high_code_mapping_edit, x_medium_code_mapping_edit, x_low_code_mapping_edit), dim=1)
        y_hat_first = decoder(x_high_code_edit, return_latents=True)
        
        x_high_code_tensor_edit, x_medium_code_tensor_edit, x_low_code_tensor, _, _, _ = encoder(face_pool_256(y_hat_first), face_pool_256(y_hat_first))
        codes = torch.cat((x_high_code_tensor_edit, x_medium_code_tensor_edit, x_low_code_tensor), dim=1)
        y_hat = decoder(codes, return_latents=True)
        '''
        # self reconstruction loss 
        x_high_code_mapping_edit_self = x_high_code_tensor + mapping_net_high(x_high_code_tensor, identity_x)
        x_medium_code_mapping_edit_self = x_medium_code_tensor + mapping_net_medium(x_medium_code_tensor, identity_x)
        x_low_code_mapping_edit_self = x_low_code_tensor + mapping_net_low(x_low_code_tensor, identity_x)
        x_self_high_code_edit = torch.cat((x_high_code_mapping_edit_self, x_medium_code_mapping_edit_self, x_low_code_mapping_edit_self), dim=1)

        x_self_recon = decoder(x_self_high_code_edit, return_latents=True)
        
        # Discriminator
        requires_grad(discriminator, True)
        with torch.no_grad():
            real_w_x = torch.cat((x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor), dim=1)
            real_w_y = torch.cat((y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor), dim=1)
            fake_w_1 = x_high_code_edit
            
        real_pred = (discriminator(real_w_x) + discriminator(real_w_y))/2
        fake_pred = discriminator(fake_w_1)
        
        
                
        loss_D = discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss_D)
        
        discriminator_optimizer.zero_grad()
        '''
        loss, loss_dict, id_logs = calc_loss(x, y, y_hat, global_step)
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # loss_D.backward()
        # discriminator_optimizer.step()
        
        # Logging related
        if global_step % opts.image_interval == 0 or (global_step < 1000 and global_step % 25 == 0):
            parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
        if global_step % opts.board_interval == 0:
            print_metrics(loss_dict, prefix='train')
            log_metrics(loss_dict, prefix='train')

        # Validation related
        val_loss_dict = None
        if global_step % opts.val_interval == 0 or global_step == opts.max_steps:
            val_loss_dict = validate()
            if val_loss_dict and (best_val_loss is None or val_loss_dict['loss'] < best_val_loss):
                best_val_loss = val_loss_dict['loss']
                checkpoint_me(val_loss_dict, is_best=True)
            
        if global_step % 20000 == 0 or global_step ==opts.max_steps:
            torch.save(encoder.state_dict(), 'output/model_based_on_pSp_encoder%d.pth'%global_step)
            torch.save(decoder.state_dict(), 'output/model_based_on_pSp_decoder%d.pth'%global_step)
            torch.save(mapping_net_high.state_dict(), 'output/model_based_on_pSp_mapping_net_high_level%d.pth'%global_step)
            torch.save(mapping_net_medium.state_dict(), 'output/model_based_on_pSp_mapping_net_medium_level%d.pth'%global_step)
            torch.save(mapping_net_low.state_dict(), 'output/model_based_on_pSp_mapping_net_low_level%d.pth'%global_step)
            ##torch.save(mapping_net_medium_level.state_dict(), 'output/model_based_on_pSp_mapping_net_medium_level%d.pth'%global_step)
            

        if global_step % opts.save_interval == 0 or global_step == opts.max_steps:     
            if val_loss_dict is not None:
                checkpoint_me(val_loss_dict, is_best=False)
            else:
                checkpoint_me(loss_dict, is_best=False)

        if global_step == opts.max_steps:
            print('OMG, finished training!')
            break

        global_step += 1
