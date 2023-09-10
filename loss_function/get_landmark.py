# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 15:48:10 2021

@author: LanGuipeng
"""
import cv2
import torch
import dlib
from torchvision.utils import make_grid
import torch.nn as nn

class LandmarkLoss(nn.Module):
    def __init__(self):
        super(LandmarkLoss, self).__init__()
        print('Loading dlib landmark194')
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("/usr/local/tarfile/lan/my_code/20220318version6/pretained_model/shape_predictor_194_face_landmarks.dat")
        self.loss = torch.nn.L1Loss()
        
    def forward(self, x, y):
        n_samples_x = x.shape[0]
        n_samples_y = y.shape[0]
        pt_pos_cat_x_middle = []
        pt_pos_cat_y_middle = []
        pt_pos_cat_x = []
        pt_pos_cat_y = []
        x_face = []
        i=0

        for i in range(n_samples_x):
            real_img = 0.5*(x[i]+1.0)
            grid_img = make_grid(real_img.data, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
            image = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray, 1)
            x_face.append(len(dets))
            for face in dets:
                if i == 0:
                    shape = self.predictor(image, face)
                    pt_pos = []
                    for pt in shape.parts():
                        pt_tuple2list = list((pt.x, pt.y))
                        pt_pos.append(pt_tuple2list)
                    pt_pos_cat_x_middle.append(pt_pos)
                else:
                    pass
                i+=1
            pt_pos_cat_x.append(pt_pos_cat_x_middle)
        landmark_for_loss_x = torch.tensor(pt_pos_cat_x)
        i=0
        for i in range(n_samples_y):
            real_img = 0.5*(y[i]+1.0)
            grid_img = make_grid(real_img.data, nrow=8, padding=2, pad_value=0,normalize=False, range=None, scale_each=False)
            image = grid_img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray, 1)
            x_face.append(len(dets))
            for face in dets:
                if i == 0:
                    shape = self.predictor(image, face)
                    pt_pos = []
                    for pt in shape.parts():
                        pt_tuple2list = list((pt.x, pt.y))
                        pt_pos.append(pt_tuple2list)
                    pt_pos_cat_y_middle.append(pt_pos)
                else:
                    pass
            pt_pos_cat_y.append(pt_pos_cat_y_middle)
        landmark_for_loss_y = torch.tensor(pt_pos_cat_y)
        
        if 0 in x_face:
            loss = 0
        else:
            loss = self.loss(torch.Tensor.float(landmark_for_loss_x), torch.Tensor.float(landmark_for_loss_y))
            count = x.shape[0]
            loss = loss / count
            
        return loss