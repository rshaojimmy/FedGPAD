#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn, autograd
import numpy as np
import random
from sklearn import metrics
from misc.utils import get_inf_iterator
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import DataParallel
from pdb import set_trace as st


class DiffLoss(nn.Module):

    def __init__(self, args):
        super(DiffLoss, self).__init__()
        self.args = args

    def forward(self, input1, input2):
        
        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach() 
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + self.args.eps)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + self.args.eps)

        diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return diff_loss


def get_lr(args, epoch):
    if epoch <= args.first_decay:
        return args.lr
    elif epoch <= args.second_decay:
        return args.lr * 0.1
    else:
        return args.lr * 0.01


class LocalUpdate(object):
    def __init__(self, args, epoch, step, images, depths, labels, images_resize):
        self.args = args
        self.loss_rec = nn.MSELoss().cuda()
        self.loss_dep = nn.MSELoss().cuda()
        self.loss_cls = nn.BCEWithLogitsLoss().cuda()
        self.loss_diff = DiffLoss(self.args).cuda()

        self.cur_epoch = epoch
        self.cur_step = step
        self.images = images
        self.depths = depths
        self.labels = labels
        self.images_resize = images_resize

    def train(self, FeatExt, FeatExt_specific, Clsfier, DepthEst, Decoder):
        FeatExt.train()
        FeatExt_specific.train()
        Clsfier.train()
        DepthEst.train()
        Decoder.train()

        # train and update
        optimizer = optim.Adam(list(FeatExt.parameters())
                              +list(FeatExt_specific.parameters())
                              +list(Clsfier.parameters())
                              +list(DepthEst.parameters()) 
                              +list(Decoder.parameters()), 
                              lr=self.args.lr)

        optimizer.zero_grad()
        catfeat, ebdfeat = FeatExt(self.images)
        _, ebdfeat_specific = FeatExt_specific(self.images)

        pred_dep = DepthEst(catfeat)
        pred_cls = Clsfier(ebdfeat)

        if self.args.concat_operation in ['add', 'addrelu'] :
            feat_union = ebdfeat + ebdfeat_specific
        elif self.args.concat_operation in ['cat', 'catrelu'] :          
            feat_union = torch.cat([ebdfeat, ebdfeat_specific], 1)  

        images_rec = Decoder(feat_union)

        loss_cls = self.loss_cls(pred_cls.squeeze(), self.labels)
        loss_dep = self.loss_dep(pred_dep, self.depths)
        loss_rec = self.loss_rec(images_rec, self.images_resize)

        loss_diff = self.loss_diff(ebdfeat, ebdfeat_specific)
        loss = loss_cls + self.args.w_dep*loss_dep + self.args.w_diff*loss_diff + self.args.w_rec*loss_rec

        loss.backward()
        optimizer.step()

        if self.cur_step % self.args.log_step == 0:
                print('Normal Update Epoch: {} Step: {}\tLoss_cls: {:.6f} Loss_dep: {:.6f}  Loss_diff: {:.6f} Loss_rec: {:.6f} featinv: {:.6f} featspe: {:.6f}'.format(
                    self.cur_epoch, self.cur_step, loss_cls.item(), loss_dep.item(), loss_diff.item(), loss_rec.item(), ebdfeat.mean(), ebdfeat_specific.mean()))


        return FeatExt.state_dict(), \
                FeatExt_specific.state_dict(), \
                Clsfier.state_dict(), \
                DepthEst.state_dict(), \
                Decoder.state_dict(), \
                loss

