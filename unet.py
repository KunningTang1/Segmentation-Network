import torch
import torch.nn as nn

def double_conv(in_planes, out_planes):
    conv = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
        # nn.Dropout(0.2),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(True),
    )
    return conv

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        
        self.downconv1 = double_conv(3, 64)
        self.maxpool = nn.MaxPool2d(2 ,2)
        
        self.downconv2 = double_conv(64, 128)
        self.downconv3 = double_conv(128, 256)
        self.downconv4 = double_conv(256, 512)

        
        self.updeconv2 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.upconv2 = double_conv(512, 256)
        
        self.updeconv3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.upconv3 = double_conv(256, 128)        
        
        self.updeconv4 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.upconv4 = double_conv(128, 64)
        
        self.out = nn.Conv2d(64, 6, 1)  # 6 is the number of classes need to be segment
        
        # Weight initialization
        for m in self.modules(): 
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            # kaiming
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        
        # encoder
        x1 = self.downconv1(x)
        x2 = self.maxpool(x1)
        
        x3 = self.downconv2(x2)
        x4 = self.maxpool(x3)
        
        x5 = self.downconv3(x4)
        x6 = self.maxpool(x5)
        
        x7 = self.downconv4(x6)
        
        x = self.updeconv2(x7)
        # y5 = crop_fun(x5, x)
        x = self.upconv2(torch.cat([x, x5],1))
        
        x = self.updeconv3(x)
        # y3 = crop_fun(x3, x)
        x = self.upconv3(torch.cat([x, x3],1))
        
        x = self.updeconv4(x)
        # y1 = crop_fun(x1, x)
        x = self.upconv4(torch.cat([x, x1],1))
        
        x = self.out(x)
        
        
        return x
    
def unet():
    net = Unet()
    return net