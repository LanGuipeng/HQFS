#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: LanGuipeng

PyTorch = 1.10
"""

import torch.nn as nn
# mapper
class mapping_network(nn.Module):
    def __init__(self, num):
        super(mapping_network, self).__init__()
        
        self.num = num
        self.mapper_block_1 = mapper_block(num).cuda()
        self.mapper_block_2 = mapper_block(num).cuda()
        self.mapper_block_3 = mapper_block(num).cuda()
        self.mapper_block_4 = mapper_block(num).cuda()
        self.mapper_block_5 = mapper_block(num).cuda()
        
    def forward(self, x, identity):
        
        x = self.mapper_block_1(x, identity)
        x = self.mapper_block_2(x, identity)
        x = self.mapper_block_3(x, identity)
        x = self.mapper_block_4(x, identity)
        x = self.mapper_block_5(x, identity)
        return x
        
class mapper_block(nn.Module):
    def __init__(self, num):
        super(mapper_block, self).__init__()
        
        self.num = num
        self.linear_layer = nn.Linear(512, 512)
        self.modulation_model = modulation_part(num).cuda()
        self.activate = nn.LeakyReLU()
    def forward(self, x, identity):
        x = self.linear_layer(x)
        x = self.modulation_model(x, identity)
        x = self.activate(x)
        return x
    

class modulation_part(nn.Module):
    def __init__(self, num):
        super(modulation_part, self).__init__()
        
        self.norm = nn.LayerNorm([num,512])
        self.f_add = nn.Sequential(nn.Linear(512,512), nn.Linear(512,512), nn.LayerNorm([1,512]), nn.LeakyReLU())
        self.f_mul = nn.Sequential(nn.Linear(512,512), nn.Linear(512,512), nn.LayerNorm([1,512]), nn.LeakyReLU())
    
    def forward(self, x, identity):
        x_norm = self.norm(x)        
        x_norm_add = self.f_add(identity)
        x_norm_mul = self.f_mul(identity)
        
        out = x_norm+x_norm_add+x_norm_mul
        return out