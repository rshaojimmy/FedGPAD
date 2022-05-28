import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
import torch
from torch import nn


def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):    
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias)

class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        # self.conv = conv3x3(in_channels, out_channels)
        self.conv = nn.Sequential(
                    conv3x3(in_channels, out_channels),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x    


class Downconv(nn.Module):
    """
    A helper Module that performs 3 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """
    def __init__(self, in_channels, out_channels):
        super(Downconv, self).__init__()

        self.downconv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            conv3x3(128, 196),
            nn.BatchNorm2d(196),
            nn.ReLU(inplace=True),

            conv3x3(196, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.downconv(x)
        return x



class DOWN(nn.Module):
    def __init__(self, in_channels, out_channels, ):
        super(DOWN, self).__init__()
        self.mpconv = nn.Sequential(
            Downconv(in_channels, out_channels),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x



class Decoder(nn.Module):
    def __init__(self, concat_operation):
        super(Decoder, self).__init__()

        if concat_operation in ['add']:
            self.fc1 = nn.Linear(512, 512*4*4, bias=False)
        elif concat_operation is 'cat': 
            self.fc1 = nn.Linear(1024, 512*4*4, bias=False)
        elif concat_operation is 'addrelu':
            self.fc1 = nn.Sequential(
                         nn.Linear(512, 512*4*4, bias=False),
                         nn.ReLU(inplace=True),
                         )
        elif concat_operation is 'catrelu': 
            self.fc1 = nn.Sequential(
                         nn.Linear(1024, 512*4*4),
                         nn.BatchNorm1d(512*4*4),
                         nn.ReLU(inplace=True),
                         )


        self.Deconv = nn.Sequential(
            nn.ConvTranspose2d(   512,      512, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),

            nn.ConvTranspose2d(   512,      256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),  

            nn.ConvTranspose2d(   256,      128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),   

            nn.ConvTranspose2d(   128,      64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True), 

            nn.ConvTranspose2d(   64,      32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True), 

            nn.ConvTranspose2d(   32,      6, 4, stride=2, padding=1, bias=False),
            nn.Tanh() 
        )

    def forward(self, x):
        x = self.fc1(x)
        x = x.resize(x.shape[0], 512, 4, 4)
        out = self.Deconv(x)
        return out

class FeatExtractor(nn.Module):
    def __init__(self, in_channels=6):
        super(FeatExtractor, self).__init__()  

        self.inc = inconv(in_channels, 64)

        self.down1 = DOWN(64, 128)
        self.down2 = DOWN(128, 128)
        self.down3 = DOWN(128, 128)

        self.embeder = nn.Sequential(
            conv3x3(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(128, 256),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            conv3x3(256, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )

        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):

        dx1 = self.inc(x)
        dx2 = self.down1(dx1)
        dx3 = self.down2(dx2)
        dx4 = self.down3(dx3)

        re_dx2 = F.adaptive_avg_pool2d(dx2, 32)
        re_dx3 = F.adaptive_avg_pool2d(dx3, 32)
        catfeat = torch.cat([re_dx2, re_dx3, dx4],1)

        out = self.embeder(dx4)
        out = self.avgpooling(out)
        out = out.view(out.size(0), -1)

        return catfeat, out


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()  

        self.classifier = nn.Linear(512, 1)  

    def forward(self, x):
        out = self.classifier(x)

        return out


class DepthEstmator(nn.Module):
    def __init__(self, in_channels=384, out_channels=1):
        super(DepthEstmator, self).__init__()

        self.conv = nn.Sequential(
            conv3x3(in_channels, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            conv3x3(128, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),            
            
            conv3x3(64, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True) 
        )


    def forward(self, x):
        x = self.conv(x)
        return x


