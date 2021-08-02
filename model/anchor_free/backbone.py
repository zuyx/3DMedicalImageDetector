
import torch
from torch import nn


class PostRes(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = nn.Conv3d(n_in, n_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(n_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(n_out, n_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(n_out)
        #self.att = CoordAtt(n_out,n_out)

        if stride != 1 or n_out != n_in:
            self.shortcut = nn.Sequential(
                nn.Conv3d(n_in, n_out, kernel_size=1, stride=stride),
                nn.BatchNorm3d(n_out))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.att(out)

        out += residual
        out = self.relu(out)
        return out

class res_unet(nn.Module):
    def __init__(self,n_classes = 1):
        super(res_unet, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):# 0，1
            blocks = []
            for j in range(num_blocks_back[i]): # 0，1，2
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i])) # （）
                    # i=0,j=0 (inc = 64 + 64 + 3, outc = 128)
                    # i=1,j=0 (inc = 64 + 64, outc = 64)
                else:
                    # i=1,j=1 (inc = 64,outc = 64)
                    # i=1,j=2 (inc = 64,outc = 64)
                    # i=0,j=1 (inc = 128,outc = 128)
                    # i=0,j=2 (inc = 128,outc = 128)
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            # back2: (inc = 131,outc = 128)
            # back3: (inc = 128,outc = 64)
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

        self.heatmap = nn.Sequential(
            nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, n_classes, kernel_size=1)
        )

        self.offset = nn.Sequential(
            nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, 3, kernel_size=1)
        )

        self.diam = nn.Sequential(
            nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(64, 1, kernel_size=1)
        )

    def forward(self, x, coord):

        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 96

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 96+96

        rev2 = self.path2(comb3)
        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))  # 64+64
        fmap = self.drop(comb2)

        heatmap = self.heatmap(fmap)
        offset = self.offset(fmap)
        diam = self.diam(fmap)

        return fmap,heatmap,offset,diam



import math
class res18d(nn.Module):
    def __init__(self,n_classes = 1):
        super(res18d, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True),
            nn.Conv3d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace=True))

        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2, stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2, stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
        self.drop = nn.Dropout3d(p=0.5, inplace=False)

        self.nodule_output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                           nn.ReLU(),
                                           # nn.Dropout3d(p = 0.3),
                                           nn.Conv3d(64, n_classes, kernel_size=1))

        self.regress_output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                            nn.ReLU(),
                                            # nn.Dropout3d(p = 0.3),
                                            nn.Conv3d(64, 3 , kernel_size=1))

        self.diam = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size=1),
                                            nn.ReLU(),
                                            # nn.Dropout3d(p = 0.3),
                                            nn.Conv3d(64, 1 , kernel_size=1))

        focal_bias = -math.log((1.0 - 0.01) / 0.01)
        self._modules['nodule_output'][2].bias.data.fill_(focal_bias)

    def forward(self, x, coord):

        out = self.preBlock(x)  # 16
        out_pool, indices0 = self.maxpool1(out)
        out1 = self.forw1(out_pool)  # 32
        out1_pool, indices1 = self.maxpool2(out1)
        out2 = self.forw2(out1_pool)  # 64
        out2_pool, indices2 = self.maxpool3(out2)
        out3 = self.forw3(out2_pool)  # 96
        out3_pool, indices3 = self.maxpool4(out3)
        out4 = self.forw4(out3_pool)  # 96

        rev3 = self.path1(out4)
        comb3 = self.back3(torch.cat((rev3, out3), 1))  # 96+96
        rev2 = self.path2(comb3)

        comb2 = self.back2(torch.cat((rev2, out2, coord), 1))  # 64+64
        comb2 = self.drop(comb2)
        fmap = comb2
        nodule_out = self.nodule_output(comb2)
        regress_out = self.regress_output(comb2)
        diam_out = self.diam(comb2)
        return fmap,nodule_out,regress_out,diam_out