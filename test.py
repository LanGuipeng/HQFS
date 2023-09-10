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
import numpy as np
from torchvision.utils import save_image

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
# -----------------------------------------Training---------------------------------------- #
encoder.cuda()
decoder.cuda()
mapping_net_high.cuda()
mapping_net_medium.cuda()
mapping_net_low.cuda()

encoder_path = 'pretained_model/model_based_on_pSp_encoder300000.pth'
decoder_path = 'pretained_model/model_based_on_pSp_decoder300000.pth'
mapping_net_high_path = 'output/model_based_on_pSp_mapping_net_high_level60000.pth'
mapping_net_medium_path = 'output/model_based_on_pSp_mapping_net_medium_level60000.pth'
mapping_net_low_path = 'output/model_based_on_pSp_mapping_net_low_level60000.pth'

encoder.load_state_dict(torch.load(encoder_path))
decoder.load_state_dict(torch.load(decoder_path))
mapping_net_high.load_state_dict(torch.load(mapping_net_high_path))
mapping_net_medium.load_state_dict(torch.load(mapping_net_medium_path))
mapping_net_low.load_state_dict(torch.load(mapping_net_low_path))

encoder.eval()
decoder.eval()
mapping_net_high.eval()
mapping_net_medium.eval()
mapping_net_low.eval()
print('model has already been loaded!')
while global_step < opts.max_steps:
    for batch_idx, batch in enumerate(train_dataloader):
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
        
            
        fake_faceB_save = 0.5 * (x.data + 1.0)
        fake_faceAB_save = 0.5 * (y.data + 1.0)
        real_faceA_save = 0.5 * (y_hat.data + 1.0)

        save = np.concatenate((fake_faceB_save.data.cpu(), fake_faceAB_save.data.cpu(),real_faceA_save.data.cpu()),axis = 3)
        save_image(torch.from_numpy(save), 'test/result_%d.png'%batch_idx)
     
        print('image%d_is_finished.png'%batch_idx)

        global_step += 1
