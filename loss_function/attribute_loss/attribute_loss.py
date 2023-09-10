# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 21:34:10 2021

@author: LanGuipeng
"""

from loss_function.attribute_loss.xception import xception
import torch.nn as nn
import torch.nn.functional as F

model = xception().cuda().eval()

def attribute_loss(x, y):
    losses = 0
    count = 0
    loss = nn.L1Loss()

    x_trans = F.interpolate(x, size=229, mode='bilinear', align_corners=False)
    y_trans = F.interpolate(y, size=229, mode='bilinear', align_corners=False)
    
    result_x_trans = model(x_trans)
    result_y_trans = model(y_trans)
    
    for x,y in zip(result_x_trans, result_y_trans):
        losses += loss(x,y)
        count += 1
    return losses/count



