import torch
import torch.nn as nn

def unet_conv(in_planes, out_planes):
    conv = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(False),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(False),
    )
    return conv


class Uresnet(nn.Module):
    def __init__(self, input_nbr = 3,label_nbr = 6):
        super(Uresnet, self).__init__()
        
        # forwarf
        self.downconv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(False),
        )      # No.1 long skip 
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 1),
            nn.ReLU(False),
        )      # No1 resudual block
        
        self.downconv3 = unet_conv(128, 128) # No2 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv4 = nn.Sequential(
            nn.Conv2d(128, 256, 1),
            nn.ReLU(False),
        )      # No2 resudual block
        
        self.downconv5 = unet_conv(256, 256) # No3 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv6 = nn.Sequential(
            nn.Conv2d(256, 512, 1),
            nn.ReLU(False),
        )      # No3 resudual block
        
        self.downconv7 = unet_conv(512, 512) # No4 long skip

        
        self.updeconv2 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
        )
           
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv4 = unet_conv(256, 256)
        
        self.updeconv3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(256, 128, kernel_size=1),
            nn.BatchNorm2d(128),
        )
           
        self.upconv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv6 = unet_conv(128, 128)
        self.updeconv4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64)
        )
        
        self.upconv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(False),
        
        )       # No6 resudual block
        self.upconv8 = unet_conv(64, 64)
        
        self.last = nn.Conv2d(64, label_nbr, 1) 
        
        
    def forward(self, x):
        
        # encoding
        x1 = self.downconv1(x) 
        x2 = self.maxpool(x1)     
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)      
        x4 += x3
        x5 = self.maxpool(x4)
        
        x6 = self.downconv4(x5)
        x7 = self.downconv5(x6)
        x7 += x6
        x8 = self.maxpool(x7)
        
        x9 = self.downconv6(x8)
        x10 = self.downconv7(x9)
        x10 += x9

        y3 = nn.functional.interpolate(x10, mode='bilinear', scale_factor=2)
        y4 = self.updeconv2(y3)
        y5 = self.upconv3(torch.cat([y4, x7],1))
        y6 = self.upconv4(y5)
        y6 += y5
        
        y6 = nn.functional.interpolate(y6, mode='bilinear', scale_factor=2)
        y7 = self.updeconv3(y6)   
        y8 = self.upconv5(torch.cat([y7, x4],1))
        y9 = self.upconv6(y8)
        y9 += y8
        
        y9 = nn.functional.interpolate(y9, mode='bilinear', scale_factor=2)
        y10= self.updeconv4(y9)
        y11 = self.upconv7(torch.cat([y10, x1],1))
        y12 = self.upconv8(y11)
        y12 += y11
     
        out = self.last(y12)
        
        return out

def uresnet():
    net = Uresnet()
    return net

