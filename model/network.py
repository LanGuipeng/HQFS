#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: LanGuipeng

PyTorch = 1.10
"""

import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import math
from model.encoder import GradualStyleEncoder
from model.decoder import Generator
from model.mapper import mapping_network


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.n_styles = 16
        # Define architecture
        self.encoder = GradualStyleEncoder()
        # Load weights if needed
        self.mask = nn.Sequential(nn.Conv1d(self.n_styles,self.n_styles,1), nn.InstanceNorm1d(self.n_styles), nn.Sigmoid())
        self.load_weights()

    def load_weights(self):
        print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load('pretained_model/model_ir_se50.pth')
        self.encoder.load_state_dict(encoder_ckpt, strict=False)

    def forward(self, x, y):
        x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor = self.encoder(x, y)

        return x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Define architecture
        self.decoder = Generator(512, 512, 8, channel_multiplier=2)
        # Load weights if needed
        self.load_weights()

    def load_weights(self):
        print('Loading decoder weights from pretrained!')
        ckpt = torch.load('/data/lan/faceswap_my_code/my_code/20220318version6/pretained_model/stylegan2-ffhq-config-f.pt')
        self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
        self.__load_latent_avg(ckpt, repeat=2*int(math.log(512, 2))-2)

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
            self.latent_avg = ckpt['latent_avg'].cuda()
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

class Network(nn.Module):
    def __init__(self, output_size, stylegan_size):
        super(Network, self).__init__()
        
        self.encoder = Encoder()
        # mapper
        self.mapper = mapping_network(16)
        # decoder
        self.decoder = Decoder()
        # self.load_weights()
    '''
    def load_weights(self):
        print('Loading encoders weights from irse50!')
        encoder_ckpt = torch.load('./pretained_model/model_based_on_pSp_encoder300000.pth')
        self.encoder.load_state_dict(encoder_ckpt, strict=False)
        
        print('Loading decoder weights from pretrained!')
        ckpt = torch.load('./pretained_model/model_based_on_pSp_decoder300000.pt')
        self.decoder.load_state_dict(ckpt, strict=False)
    '''    
    
    def forward(self, x, y, identity,
                resize=True, latent_mask=None, input_code=False, randomize_noise=True, inject_latent=None, return_latents=False, alpha=None):
        
        x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor = self.encoder(x, y)
        x_code_edit = torch.cat((x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor), dim=1)+ self.mapper(torch.cat((x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor), dim=1), identity)
        input_is_latent = not input_code
        images, result_latent = self.decoder([x_code_edit],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)
        return images, result_latent

