import torch
from torch import nn
import Resnet
import torch.nn.functional as F
from functools import partial

__all__ = ['Resnet50_UNet', 'VGG_UNet']

# VGG_Unet网络结构

def double_conv(in_channels, out_channels):  # 双层卷积模型
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        # in_channels指输入通道的深度即输入图像的深度（一般RGB为3）；in_channels指输出通道的深度即卷积核个数，也为下一个卷积的输入深度；3指kernel_size，即卷积核3*3\
        ##若输入数据矩阵尺寸：W1×H1×D1
        ##若输入数据矩阵尺寸：W2×H2×D2
        ##特征图组尺寸计算如下：W2=(W1-F+2P)/S+1      F-卷积核尺寸
        ##                                        S-卷积步长
        ##                : H2=(W1-F+2P)/S+1      P-零填充数量
        ##                  D2=K                  K-卷积核个数
        nn.BatchNorm2d(out_channels),  # 加入Bn层提高网络泛化能力（防止过拟合），加收敛速度
        ##BN层即批归一化，调整权值分布使得输出与输入具有相同的分布，经常插入到全连接层后，非线性激活前
        ##去均值除方差
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out



class VGG_UNet(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock(input_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = VGGBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = VGGBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = VGGBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = VGGBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = VGGBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = VGGBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        if self.deep_supervision:
            self.final1 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)


    def forward(self, input):
        # print('input:',input.shape)
        x0_0 = self.conv0_0(input) #2,32,256,256
        # print('x0_0:',x0_0.shape)#2,64,128,128
        x1_0 = self.conv1_0(self.pool(x0_0))
        # print('x1_0:',x1_0.shape)
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))#2,32,256,256
        # print('x0_1:',x0_1.shape)

        x2_0 = self.conv2_0(self.pool(x1_0))#2,128,64,64
        # print('x2_0:',x2_0.shape)
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))#2,64,128,128
        # print('x1_1:',x1_1.shape)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))#2,32,256,256
        # print('x0_2:',x0_2.shape)

        x3_0 = self.conv3_0(self.pool(x2_0))#2,256,32,32
        # print('x3_0:',x3_0.shape)
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))#2,128,64,64
        # print('x2_1:',x2_1.shape)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))#2,64,128,128
        # print('x1_2:',x1_2.shape)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))#2,32,256,256
        # print('x0_3:',x0_3.shape)
        x4_0 = self.conv4_0(self.pool(x3_0))#2,512,16,16
        # print('x4_0:',x4_0.shape)
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))#2,256,32,32
        # print('x3_1:',x3_1.shape)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))#2,128,64,64
        # print('x2_2:',x2_2.shape)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))#2,64,128,128
        # print('x1_3:',x1_3.shape)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))#2,32,256,256
        # print('x0_4:',x0_4.shape)

        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            ##根据代码nn.Conv2d(nb_filter[0], num_classes, kernel_size=1)，可以确定卷积核的大小为1x1，
            # 这意味着输入的特征图张量在空间维度上不会发生变化。但是，由于输出通道数为num_classes，因此输出特征图张量的通道数将变为num_classes，
            # 其他维度保持不变。具体而言，假设输入特征图的大小为[batch_size, nb_filter[0], H, W]，则输出特征图的大小为[batch_size, num_classes, H, W]。
            output = self.final(x0_4)#2,1,256,256
            return output




# Resnet_Unet网络结构
nonlinearity = partial(F.relu,inplace=True)

class Dblock_more_dilate(nn.Module):
    def __init__(self, channel):
        super(Dblock_more_dilate, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out + dilate5_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x



class Resnet50_UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(Resnet50_UNet, self).__init__()

        filters = [256, 512, 1024, 2048]
        resnet = Resnet.resnet50(pretrained=True)
        print(resnet)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock_more_dilate(2048)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)
        self.final = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x) #2,3,256,256
        x = self.firstbn(x)  #2,64,128,128
        x = self.firstrelu(x)#2,64,128,128
        x = self.firstmaxpool(x)#2,64,64,64
        e1 = self.encoder1(x)#2,64,64,64
        e2 = self.encoder2(e1)#2,512,32,32
        e3 = self.encoder3(e2)#2,1024,16,16
        e4 = self.encoder4(e3)#2,2048,8,8

        # Center
        e4 = self.dblock(e4)#2,64,64,64

        # Decoder
        d4 = self.decoder4(e4) + e3  #2,1024,16,16
        d3 = self.decoder3(d4) + e2  #2,512,32,32
        d2 = self.decoder2(d3) + e1  #2,256,64,64
        d1 = self.decoder1(d2)  #2,256,128,128
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)#2,3,256,256

        out=self.final(out)
        return out
        # return F.sigmoid(out)