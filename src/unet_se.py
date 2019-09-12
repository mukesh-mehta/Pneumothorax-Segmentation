from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
from torchvision.models import resnet34, resnet101, resnet50, resnet152
import torchvision
from torchsummary import summary
import pdb

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class DecoderSE(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(DecoderSE, self).__init__()
        self.conv1 = ConvBnRelu(in_channels, middle_channels)
        self.deconv = nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.se = SELayer(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = self.se(x)

        return x

class UNetResNetSE(nn.Module):
    """PyTorch U-Net model using ResNet(34, 101 or 152) encoder.

    UNet: https://arxiv.org/abs/1505.04597
    ResNet: https://arxiv.org/abs/1512.03385
    Proposed by Alexander Buslaev: https://www.linkedin.com/in/al-buslaev/

    Args:
            encoder_depth (int): Depth of a ResNet encoder (34, 101 or 152).
            num_classes (int): Number of output classes.
            num_filters (int, optional): Number of filters in the last layer of decoder. Defaults to 32.
            dropout_2d (float, optional): Probability factor of dropout layer before output layer. Defaults to 0.2.
            pretrained (bool, optional):
                False - no pre-trained weights are being used.
                True  - ResNet encoder is pre-trained on ImageNet.
                Defaults to False.
            is_deconv (bool, optional):
                False: bilinear interpolation is used in decoder.
                True: deconvolution is used in decoder.
                Defaults to False.

    """

    def __init__(self, encoder_depth, num_classes=1, num_filters=32, dropout_2d=0.2,
                 pretrained=True, is_deconv=True):
        super(UNetResNetSE, self).__init__()
        #pdb.set_trace()
        self.name = 'UNetResNetSE_'+str(encoder_depth)
        self.num_classes = num_classes
        self.dropout_2d = dropout_2d

        if encoder_depth == 34:
            self.encoder = torchvision.models.resnet34(pretrained=pretrained)
            bottom_channel_nr = 512
        elif encoder_depth == 50:
            self.encoder = torchvision.models.resnet50(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 101:
            self.encoder = torchvision.models.resnet101(pretrained=pretrained)
            bottom_channel_nr = 2048
        elif encoder_depth == 152:
            self.encoder = torchvision.models.resnet152(pretrained=pretrained)
            bottom_channel_nr = 2048
        else:
            raise NotImplementedError('only 34, 101, 152 version of Resnet are implemented')

        self.pool = nn.MaxPool2d(2, 2)

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(self.encoder.conv1,
                                   self.encoder.bn1,
                                   self.encoder.relu)
                                   #self.pool)
        self.se1 = SELayer(num_filters*2)
        self.conv2 = self.encoder.layer1
        self.se2 = SELayer(num_filters*8)
        self.conv3 = self.encoder.layer2
        self.se3 = SELayer(num_filters*16)
        self.conv4 = self.encoder.layer3
        self.se4 = SELayer(num_filters*32)

        self.conv5 = self.encoder.layer4
        self.se5 = SELayer(num_filters*64)

        self.center = DecoderSE(bottom_channel_nr, num_filters * 8 * 2, num_filters * 8)
        self.dec5 = DecoderSE(bottom_channel_nr + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec4 = DecoderSE(bottom_channel_nr // 2 + num_filters * 8, num_filters * 8 * 2, num_filters * 8)
        self.dec3 = DecoderSE(bottom_channel_nr // 4 + num_filters * 8, num_filters * 4 * 2, num_filters * 2)
        self.dec2 = DecoderSE(bottom_channel_nr // 8 + num_filters * 2, num_filters * 2 * 2, num_filters * 2 * 2)
        self.dec1 = DecoderSE(num_filters * 2 * 2, num_filters * 2 * 2, num_filters)
        self.dec0 = ConvBnRelu(num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)
        #self.classifier = nn.Linear(num_filters * 256 * 256, 1)

    def forward(self, x):
        conv1 = self.conv1(x) #;print(conv1.size())
        se1   = self.se1(conv1)
        conv2 = self.conv2(se1) #;print(conv2.size())
        se2   = self.se2(conv2)
        conv3 = self.conv3(se2) #;print(conv3.size())
        se3   = self.se3(conv3)
        conv4 = self.conv4(se3) #;print(conv4.size())
        se4   = self.se4(conv4)
        conv5 = self.conv5(se4) #;print(conv5.size())
        se5   = self.se5(conv5)

        pool = self.pool(se5)
        center = self.center(pool)

        dec5 = self.dec5(torch.cat([center, se5], 1))

        dec4 = self.dec4(torch.cat([dec5, se4], 1))
        dec3 = self.dec3(torch.cat([dec4, se3], 1))
        dec2 = self.dec2(torch.cat([dec3, se2], 1))
        dec1 = self.dec1(dec2)
        dec0 = self.dec0(dec1)
        out = self.pool(dec0)

        #cls_out = self.classifier(F.dropout(dec0.view(dec0.size(0), -1), p=0.25))

        return self.final(F.dropout2d(out, p=self.dropout_2d)), None
    
    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def get_params(self, base_lr):
        group1 = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
        group2 = [self.dec0, self.dec1, self.dec2, self.dec3, self.dec4, self.dec5, self.center, self.se1, self.se2, self.se3, self.se4, self.se5,]
        group3 = [self.final]

        params1 = []
        for x in group1:
            for p in x.parameters():
                params1.append(p)
        
        param_group1 = {'params': params1, 'lr': base_lr / 10}

        params2 = []
        for x in group2:
            for p in x.parameters():
                params2.append(p)
        param_group2 = {'params': params2, 'lr': base_lr / 2}

        params3 = []
        for x in group3:
            for p in x.parameters():
                params3.append(p)
        param_group3 = {'params': params3, 'lr': base_lr}

        return [param_group1, param_group2, param_group3]


def test():
    model = UNetResNetSE(34).cuda()
    model.freeze_bn()
    # inputs = torch.randn(2,3,128,128).cuda()
    # out, _ = model(inputs)
    #print(model)
    # print(out.size()) #, cls_taret.size())
    #print(out)
    print(summary(model, input_size=(3, 256, 256)))

if __name__ == '__main__':
    test()