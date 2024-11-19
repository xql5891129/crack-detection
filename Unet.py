import torch
import torch.nn as nn
from torch.utils.data.datapipes.utils.decoder import Decoder
import Resnet

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


class UNetModel(torch.nn.Module):

    def __init__(self, in_features=1, out_features=2, init_features=64):
        super(UNetModel, self).__init__()
        features = init_features
        self.encode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )#64
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )#128
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )#256
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )#512
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.encode_decode_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*16, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 16),
            torch.nn.ReLU()
        )

        self.upconv4 = torch.nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        ##ConvTranspose2d即转置卷积，一种上采样方法。第一、二参数分别为in_channels, out_channels
        ##当stride>2时，进行插值操作
        # 1）在输入feature map的每2个像素之间增加(s-1)个0
        # 2）在输入的底边和右边进行padding——(c+2p-k)mod(s)，其中c的计算方式见下方。
        # 3）最后根据公式进行外围的padding。
        self.decode_layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*16, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features*8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*8, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 8),
            torch.nn.ReLU(),
        )
        self.upconv3 = torch.nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decode_layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*8, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*4, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 4),
            torch.nn.ReLU()
        )
        self.upconv2 = torch.nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decode_layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*4, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features*2, out_channels=features*2, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features * 2),
            torch.nn.ReLU()
        )
        self.upconv1 = torch.nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decode_layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features*2, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, stride=1),
            torch.nn.BatchNorm2d(num_features=features),
            torch.nn.ReLU()
        )
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=features, out_channels=out_features, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        # enc1 = self.encode_layer1(x)
        # enc2 = self.encode_layer2(self.pool1(enc1))
        # enc3 = self.encode_layer3(self.pool2(enc2))
        # enc4 = self.encode_layer4(self.pool3(enc3))


        encoder = nn.Sequential(*list(Resnet.resnet50().children())[:-2])
        encoder.cuda()

        print(encoder)


        enc=encoder[3](encoder[2](encoder[1](encoder[0](x)))  )

        print(enc.size())
        enc1=enc
        enc2 = encoder[4](enc1)
        print(enc2.size())
        enc3 = encoder[5](enc2)
        print(enc3.size())
        enc4 = encoder[6](enc3)
        print(enc4.size())



        bottleneck = self.encode_decode_layer(self.pool4(enc4))
        dec4 = self.upconv4(bottleneck)##转置卷积上采样
        print(dec4.size())
        print(enc4.size())
        dec4 = torch.cat((dec4, enc4), dim=1)##cat合并矩阵，tim=0按列拼，tim=1按行拼
        dec4 = self.decode_layer4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decode_layer3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decode_layer2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decode_layer1(dec1)

        out = self.out_layer(dec1)
        return out


