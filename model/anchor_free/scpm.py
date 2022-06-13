import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


def Conv_Block(in_planes, out_planes, stride=1):
    """3x3x3 convolution with batchnorm and relu"""
    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                                   padding=1, bias=False),
                         nn.BatchNorm3d(out_planes),
                         nn.ReLU(inplace=True))


def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Norm(nn.Module):
    def __init__(self, N):
        super(Norm, self).__init__()
        self.normal = nn.BatchNorm3d(N)

    def forward(self, x):
        return self.normal(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = Norm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = Norm(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False) # 尺寸不变
        self.bn1 = Norm(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, # stride = 2 尺寸变为一半,stride = 1尺寸不变
                               padding=1, bias=False) # 尺寸不变
        self.bn2 = Norm(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False) # 尺寸不变
        self.bn3 = Norm(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x): # skip connection
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SAC(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(SAC, self).__init__()

        self.conv_1 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_3 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv_5 = nn.Conv3d(
            input_channel, out_channel, kernel_size=3, stride=1, padding=3, dilation=3)
        self.weights = nn.Parameter(torch.ones(3))
        self.softmax = nn.Softmax(0)

    def forward(self, inputs):
        feat_1 = self.conv_1(inputs)
        feat_3 = self.conv_3(inputs)
        feat_5 = self.conv_5(inputs)
        weights = self.softmax(self.weights)
        feat = feat_1 * weights[0] + feat_3 * weights[1] + feat_5 * weights[2]
        return feat


class Pyramid_3D(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, feature_size=256, using_sac=False):
        # C2_size = 32 , C3_size = 64, C4_size = 64, C5_size = 64 feature_size = 64
        super(Pyramid_3D, self).__init__()

        self.P5_1 = nn.Conv3d(C5_size, feature_size,
                              kernel_size=1, stride=1, padding=0) # (64,64)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest') # ()
        self.P5_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) #if not using_sac else SAC(feature_size, feature_size) # (256,256)

        self.P4_1 = nn.Conv3d(C4_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) #if not using_sac else SAC(feature_size, feature_size)

        self.P3_1 = nn.Conv3d(C3_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) #if not using_sac else SAC(feature_size, feature_size)

        self.P2_1 = nn.Conv3d(C2_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv3d(feature_size, feature_size, kernel_size=3, stride=1,
                              padding=1) if not using_sac else SAC(feature_size, feature_size)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs
        # C5 size: (64,6,6,6)
        P5_x = self.P5_1(C5)  # size: (64,6,6,6)
        P5_upsampled_x = self.P5_upsampled(P5_x) # (64,12,12,12)
        P5_x = self.P5_2(P5_x) # (64,6,6,6)

        # C4 size: (64,12,12,12)
        P4_x = self.P4_1(C4) # size: (64,12,12,12)
        P4_x = P5_upsampled_x + P4_x  # 融入更抽象的featuremap 信息
        P4_upsampled_x = self.P4_upsampled(P4_x) # size: (64,24,24,24)
        P4_x = self.P4_2(P4_x) # (256,24,24,24)

        # C3 size: (64,24,24,24)
        P3_x = self.P3_1(C3) #
        P3_x = P3_x + P4_upsampled_x # (256,24,24,24)
        P3_upsampled_x = self.P3_upsampled(P3_x) # (64,48,48,48)
        P3_x = self.P3_2(P3_x) # (256,48,48,48)

        # C2 size: (32,48,48,48)
        P2_x = self.P2_1(C2) # (256,48,48,48)
        P2_x = P2_x + P3_upsampled_x
        P2_x = self.P2_2(P2_x) # (256,48,48,48)

        return [P2_x, P3_x, P4_x, P5_x] # (256,48) (256,48) (256,24) (256,12)


class Attention_SE_CA(nn.Module):
    def __init__(self, channel):
        super(Attention_SE_CA, self).__init__()
        self.Global_Pool = nn.AdaptiveAvgPool3d(1)
        self.FC1 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.ReLU(), )
        self.FC2 = nn.Sequential(nn.Linear(channel, channel),
                                 nn.Sigmoid(), )

    def forward(self, x):
        G = self.Global_Pool(x)
        G = G.view(G.size(0), -1)
        fc1 = self.FC1(G)
        fc2 = self.FC2(fc1)
        fc2 = torch.unsqueeze(fc2, 2)
        fc2 = torch.unsqueeze(fc2, 3)
        fc2 = torch.unsqueeze(fc2, 4)
        return fc2*x

cls = 1
class SCPMNet(nn.Module):

    def __init__(self, block, layers): # block BasicBlock class , layers [2, 2, 3, 3]
        self.inplanes = 32
        super(SCPMNet, self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3,
                               stride=1, padding=1, bias=False) # (1,32)
        self.bn1 = Norm(32)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3,
                               stride=1, padding=1, bias=False) # (32,32) 提高channel 的原因是什么？
        self.bn2 = Norm(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) # 下采样 放缩因子为2
        self.layer1 = self._make_layer(block, 32, layers[0])
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)
        self.atttion1 = Attention_SE_CA(32)
        self.atttion2 = Attention_SE_CA(32)
        self.atttion3 = Attention_SE_CA(64)
        self.atttion4 = Attention_SE_CA(64)
        self.conv_1 = Conv_Block(64 + 3, 64)
        self.conv_2 = Conv_Block(64 + 3, 64)
        self.conv_3 = Conv_Block(64 + 3, 64)
        self.conv_4 = Conv_Block(64 + 3, 64)
        self.conv_8x = Conv_Block(64, 64)
        self.conv_4x = Conv_Block(64, 64)
        self.conv_2x = Conv_Block(64, 64)
        self.convc = nn.Conv3d(64, 1, kernel_size=1, stride=1)
        self.convr = nn.Conv3d(64, 1, kernel_size=1, stride=1)
        self.convo = nn.Conv3d(64, 3, kernel_size=1, stride=1)
        if block == BasicBlock:
            fpn_sizes = [self.layer1[layers[0]-1].conv2.out_channels, self.layer2[layers[1]-1].conv2.out_channels,
                         self.layer3[layers[2]-1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer1[layers[0]-1].conv3.out_channels, self.layer2[layers[1]-1].conv3.out_channels,
                         self.layer3[layers[2]-1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.fpn = Pyramid_3D(
            fpn_sizes[0], fpn_sizes[1], fpn_sizes[2], fpn_sizes[3], feature_size=64)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    #
    def _make_layer(self,    block, planes, blocks, stride=1):
        '''
        self.inplanes = 32 init
        :param block: BasicBlock
        :param planes: 32 || 64 || 64 || 64
        :param blocks: 2 || 2 || 3 || 3
        :param stride: 1 || 2 || 2 || 2
        :return:
        '''
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                Norm(planes * block.expansion),
            ) # downsample(k=1,stride=2) ->得到的map的w,h,d是原来尺寸的一半
        # layer1 : block1(32,32,1,None), 2个block： block2(32,32), block3(32,32)
        # layer2 : block(32,64,2,downsample(32,64,k=1，stride = 2)), 2 个 block : block2(64,64) block3(64,64)
        # layer3 : block(64,64,2,downsample(64,64,k=1, stride = 2)), 3 个 block ： block2(64,64) block3(64,64) block4(64,64)
        # layer4 : block(64,64,2,downsample(64,64,k=1, stride = 2)), 3 个 block ： block2(64,64) block3(64,64) block4(64,64)
        # self.inplanes = 64
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x,coord=None):
        x = self.conv1(x) # x input : image (1,96,96,96)
        x = self.bn1(x)
        x = self.relu(x) # x (32,96,96,96)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x) # x (32,96,96,96)
        x = self.maxpool(x)
        x = self.atttion1(x) # x (32,48,48,48)
        x1 = self.layer1(x) # 经过 block1(32,32,1,None), block2(32,32), block3(32,32) 后x1 size（32,48,48,48）
        x1 = self.atttion2(x1)
        x2 = self.layer2(x1) # 经过 block1(32,64,2,downsample(32,64,k=1，stride = 2)), block2(64,64) block3(64,64) 后 x2 size: (64,24,24,24)
        x2 = self.atttion3(x2)
        x3 = self.layer3(x2) # 经过 block1(64,64,2,downsample(64,64,k=1, stride = 2)), block2(64,64) block3(64,64) block4(64,64) 后 x3 size (64,12,12,12)
        x3 = self.atttion4(x3)
        x4 = self.layer4(x3) # 经过 block1(64,64,2,downsample(64,64,k=1, stride = 2)), block2(64,64) block3(64,64) block4(64,64) 后 x4 size (64,6,6,6)

        feats = self.fpn([x1, x2, x3, x4])
        #feats[0] = torch.cat([feats[0], c], 1) # c_2 coord (3*48*48*48)
        #feats[0] = self.conv_1(feats[0]) # (64,48,48,48)
        #feats[1] = torch.cat([feats[1], c_4], 1) # c_4 coord (3*24*24*24)
        #feats[1] = self.conv_2(feats[1]) # (64,24,24,24)
        #feats[2] = torch.cat([feats[2], c_8], 1) # c_8 coord (3*12*12*12)
        #feats[2] = self.conv_3(feats[2]) # (64,12,12,12)
        #feats[3] = torch.cat([feats[3], c_16], 1) # c_16 coord (3*6*6*6)
        #feats[3] = self.conv_4(feats[3]) # (64,6,6,6)
        if cls == 1:
            Cls1 = self.convc(feats[0])  # (1,48,48,48)
            Reg1 = self.convr(feats[0])  #（1,48,48,48）
            Off1 = self.convo(feats[0])  # (3,48,48,48)
            return feats[0], Cls1, Off1, Reg1
        else:
            feat_8x = F.upsample(
                feats[3], scale_factor=2, mode='nearest') + feats[2]  # (64,12,12,12)
            feat_8x = self.conv_8x(feat_8x)  # (64,12,12,12)
            feat_4x = F.upsample(
                feat_8x, scale_factor=2, mode='nearest') + feats[1]  # (64,24,24,24)
            feat_4x = self.conv_4x(feat_4x)  # (64,24,24,24)
            feat_2x = F.upsample(feat_4x, scale_factor=2, mode='nearest')  # (64，48，48，48)
            feat_2x = self.conv_2x(feat_2x)  # (64,48,48,48)
            Cls2 = self.convc(feat_2x) # (1,48,48,48)
            Reg2 = self.convr(feat_2x)  # (1,48,48,48)
            Off2 = self.convo(feat_2x)  # (3,48,48,48)
            return feat_2x, Cls2, Off2, Reg2
        #output = {}
        #output['Cls1'] = Cls1
        #output['Reg1'] = Reg1
        #output['Off1'] = Off1
        #output['Cls2'] = Cls2
        #output['Reg2'] = Reg2
        #output['Off2'] = Off2

def scpmnet18(**kwargs):
    """Using ResNet-18 as backbone for SCPMNet.
    """
    model = SCPMNet(BasicBlock, [2, 2, 3, 3], **kwargs)  # [2,2,2,2]
    if cls == 1:
        print("采用Cls1 Reg1 Off1")
    else:
        print("采用Cls2 Reg2 Off2")
    return model

def get_model(**kwargs):
    model = SCPMNet(BasicBlock, [2, 2, 3, 3], **kwargs)  # [2,2,2,2]
    if cls == 1:
        print("采用Cls1 Reg1 Off1")
    else:
        print("采用Cls2 Reg2 Off2")
    return model

if __name__ == '__main__':
    device = torch.device("cuda")
    input = torch.ones(1, 1, 96, 96, 96).to(device)
    coord_2 = torch.ones(1, 3, 48, 48, 48).to(device)
    coord_4 = torch.ones(1, 3, 24, 24, 24).to(device)
    coord_8 = torch.ones(1, 3, 12, 12, 12).to(device)
    coord_16 = torch.ones(1, 3, 6, 6, 6).to(device)
    label = torch.ones(1, 3, 5).to(device)
    net = scpmnet18().to(device)
    net.eval()
    out = net(input)
    print(out)
    print('finish ')