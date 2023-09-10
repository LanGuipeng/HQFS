#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: LanGuipeng

PyTorch = 1.10
"""

import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from collections import namedtuple


# ------------------------------------------encoder------------------------------------------ #
# Using FPN 
# With mapping network
# Based on pSp
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """

def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]

def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks

class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class bottleneck_IR_SE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth)
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut

def _upsample_add(x, y):
    _, _, H, W = y.size()
    return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

class Backbone(nn.Module):
    def __init__(self, input_size, num_layers=50, drop_ratio=0.4, affine=True):
        super(Backbone, self).__init__()
        assert input_size in [112, 224], "input_size should be 112 or 224"

        blocks = get_blocks(num_layers)
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        if input_size == 112:
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(drop_ratio),
                                              Flatten(),
                                              nn.Linear(512 * 7 * 7, 512),
                                              nn.BatchNorm1d(512, affine=affine))
        else:
            self.output_layer = nn.Sequential(nn.BatchNorm2d(512),
                                              nn.Dropout(drop_ratio),
                                              Flatten(),
                                              nn.Linear(512 * 14 * 14, 512),
                                              nn.BatchNorm1d(512, affine=affine))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

def IR_SE_50(input_size):
    """Constructs a ir_se-50 model."""
    model = Backbone(input_size, num_layers=50, drop_ratio=0.4, affine=False)
    return model

def IR_SE_101(input_size):
    """Constructs a ir_se-101 model."""
    model = Backbone(input_size, num_layers=100, drop_ratio=0.4, affine=False)
    return model

def IR_SE_152(input_size):
    """Constructs a ir_se-152 model."""
    model = Backbone(input_size, num_layers=152, drop_ratio=0.4, affine=False)
    return model

class MappingNet(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(MappingNet, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x
# the size of generated images
class GradualStyleEncoder(nn.Module):
    def __init__(self, num_layers=50, stylegan_size=512):
        super(GradualStyleEncoder, self).__init__()
        blocks = get_blocks(num_layers)
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck_IR_SE(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = nn.Sequential(*modules)
        
        self.styles = nn.ModuleList()
        log_size = int(math.log(stylegan_size, 2))
        self.style_count = 2*log_size -2
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = MappingNet(512, 512, 16)
            elif i < self.middle_ind:
                style = MappingNet(512, 512, 32)
            else:
                style = MappingNet(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.medium_number = self.middle_ind - self.coarse_ind 
        self.mask_high_codes = nn.Sequential(nn.Conv1d(self.coarse_ind,self.coarse_ind,1), nn.InstanceNorm1d(self.coarse_ind), nn.Sigmoid())
        self.mask_medium_codes = nn.Sequential(nn.Conv1d(self.medium_number,self.medium_number,1), nn.InstanceNorm1d(self.medium_number), nn.Sigmoid())
        
    def forward(self, x, y):
        x = self.input_layer(x)
        y = self.input_layer(y)

        latents = []
        latentsy = []
        modulelist = list(self.body._modules.values())

        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        
        for k, ly in enumerate(modulelist):
            y = ly(y)
            if k == 6:
                cy1 = y
            elif k == 20:
                cy2 = y
            elif k == 23:
                cy3 = y
        
        x_high_code = []
        x_medium_code = []
        x_low_code = []
        # image x
        for j in range(self.coarse_ind):
            x_high_code.append(self.styles[j](c3))
            latents.append(self.styles[j](c3))

        p2 = _upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            x_medium_code.append(self.styles[j](p2))
            latents.append(self.styles[j](p2))

        p1 = _upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            x_low_code.append(self.styles[j](p1))
            latents.append(self.styles[j](p1))
        
        # image y
        y_high_code = []
        y_medium_code = []
        y_low_code = []
        for jy in range(self.coarse_ind):
            y_high_code.append(self.styles[jy](cy3))
            latentsy.append(self.styles[jy](cy3))

        py2 = _upsample_add(cy3, self.latlayer1(cy2))
        for jy in range(self.coarse_ind, self.middle_ind):
            y_medium_code.append(self.styles[jy](py2))
            latentsy.append(self.styles[jy](py2))

        py1 = _upsample_add(py2, self.latlayer2(cy1))
        for jy in range(self.middle_ind, self.style_count):
            y_low_code.append(self.styles[jy](py1))
            latentsy.append(self.styles[jy](py1))
        
        # combine latent code outside encoder
        x_high_code_tensor = torch.stack(x_high_code, dim=1)
        x_medium_code_tensor = torch.stack(x_medium_code, dim=1)
        x_low_code_tensor = torch.stack(x_low_code, dim=1)
        y_high_code_tensor = torch.stack(y_high_code, dim=1)
        y_medium_code_tensor = torch.stack(y_medium_code, dim=1)
        y_low_code_tensor = torch.stack(y_low_code, dim=1)
        return x_high_code_tensor, x_medium_code_tensor, x_low_code_tensor, y_high_code_tensor, y_medium_code_tensor, y_low_code_tensor