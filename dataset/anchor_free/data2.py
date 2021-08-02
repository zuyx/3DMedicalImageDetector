import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import sys

sys.path.append('../../../')
from dataset.crop import Crop
from dataset.augment import augment
from dataset.anchor_free.labelMapping import LabelMapping

class DataBowl3Detector(Dataset):
    def __init__(self, data_dir, patient_ids, config, phase='train', split_comber=None):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        self.max_stride = config['max_stride']
        self.stride = config['stride']  # 4
        sizelim = config['sizelim'] / config['reso']  # 2.5/1
        sizelim2 = config['sizelim2'] // config['reso']  # 10/1
        sizelim3 = config['sizelim3'] // config['reso']  # 20/1
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']  # {'flip':True,'swap':False,'scale':True,'rotate':False}
        self.pad_value = config['pad_value']
        self.split_comber = split_comber
        idcs = patient_ids  # np.load(split_path)

        if phase != 'test':
            idcs = [f for f in idcs if (f not in self.blacklist)]

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
        self.kagglenames = [f for f in self.filenames]  # if len(f.split('/')[-1].split('_')[0])>20]

        labels = []
        for idx in idcs:
            l = np.load(data_dir + idx + '_label.npy', allow_pickle=True)
            if np.all(l == 0):  # without nodule
                l = np.array([])
            labels.append(l)
        self.sample_bboxes = labels  # nodule
        if self.phase != 'test':
            self.bboxes = []
            for i, l in enumerate(labels):  # [[81.933381 166.26051683000003 43.932055500000004 6.440878725]]
                if len(l) > 0:
                    for t in l:
                        if t[3] > sizelim:  # 直径>2.5
                            self.bboxes.append([np.concatenate([[i], t])])
                        if t[3] > sizelim2:  # 直径>10
                            self.bboxes += [[np.concatenate([[i], t])]] * 2
                        if t[3] > sizelim3:  # 直径>20
                            self.bboxes += [[np.concatenate([[i], t])]] * 4
            self.bboxes = np.concatenate(self.bboxes, axis=0)  # many boxes
        self.crop = Crop(config)
        self.label_mapping = LabelMapping(config, self.phase)

    def __getitem__(self, idx, split=None):
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))  # seed according to time
        isRandomImg = False
        isRandom = False
        if self.phase != 'test' and idx >= len(self.bboxes):
            isRandom = True
            idx = idx % len(self.bboxes)
            isRandomImg = np.random.randint(2)  # 随机产生0和1

        if self.phase != 'test':
            if not isRandomImg:
                bbox = self.bboxes[
                    idx]  # num,x,y,z,r,第一个参数代表的是第几个CT序列（trainlist排列好的）后面代表的是xyz和直径。当直径大于2.5写1行，大于10写1+2行，大于20写1+2+4行
                filename = self.filenames[int(bbox[0])]  # 第int(bbox[0])序列的路径（包含名称_clearn.npy）
                imgs = np.load(filename)

                bboxes = self.sample_bboxes[
                    int(bbox[0])]  # self.sample_bboxes就是800CT对应的结节的xyz和直径，是一个列表list（可能是空，也可能是多行）
                isScale = self.augtype['scale'] and (self.phase == 'train')  # True
                # imgs是_clearn.npy导入的3d矩阵(仅仅包含肺部并且分辨率为1×1*1mm)/bbox代表[556 74.7895776 249.75681577 267.72357450000004 10.57235285],bbox[1:]代表[74.7895776 249.75681577 267.72357450000004 10.57235285]
                # bboxes代表[[74.7895776 249.75681577 267.72357450000004 10.57235285],[165.8177988 172.30634478 79.42561491000001 20.484109399999998]]等于或包含bbox[1:].
                sample, target, bboxes, coord = self.crop(imgs, bbox[1:], bboxes, isScale, isRandom)
                if self.phase == 'train' and not isRandom:
                    sample, target, bboxes, coord = augment(sample, target, bboxes, coord,
                                                            ifflip=self.augtype['flip'],
                                                            ifrotate=self.augtype['rotate'],
                                                            ifswap=self.augtype['swap'])
            else:
                randimid = np.random.randint(len(self.kagglenames))
                filename = self.kagglenames[randimid]
                imgs = np.load(filename)
                bboxes = self.sample_bboxes[randimid]
                # 围绕target裁剪，一个target对应一个样本
                sample, target, bboxes, coord = self.crop(imgs, [], bboxes, isScale=False, isRand=True)
            labels = np.zeros(len(bboxes)).astype(int)

            datadict = self.label_mapping(target, bboxes, labels)
            # 归一化
            sample = (sample.astype(np.float32) - 128) / 128
            datadict['coord'] = torch.from_numpy(coord)
            datadict['image'] = torch.from_numpy(sample)
            return datadict
        else:  # test
            imgs = np.load(self.filenames[idx])
            bboxes = self.sample_bboxes[idx]
            nz, nh, nw = imgs.shape[1:]  # (197, 181, 224)
            pz = int(np.ceil(float(nz) / self.stride)) * self.stride
            ph = int(np.ceil(float(nh) / self.stride)) * self.stride
            pw = int(np.ceil(float(nw) / self.stride)) * self.stride
            imgs = np.pad(imgs, [[0, 0], [0, pz - nz], [0, ph - nh], [0, pw - nw]], 'constant',
                          constant_values=self.pad_value)  # 能被stride整除
            xx, yy, zz = np.meshgrid(np.linspace(-0.5, 0.5, imgs.shape[1] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[2] // self.stride),
                                     np.linspace(-0.5, 0.5, imgs.shape[3] // self.stride), indexing='ij')
            coord = np.concatenate([xx[np.newaxis, ...], yy[np.newaxis, ...], zz[np.newaxis, :]], 0).astype('float32')
            imgs, nzhw = self.split_comber.split(imgs)  # 把imgs分割成n个sidelen块
            coord2, nzhw2 = self.split_comber.split(coord,
                                                    side_len=self.split_comber.side_len // self.stride,
                                                    max_stride=self.split_comber.max_stride // self.stride,
                                                    margin=self.split_comber.margin // self.stride)
            assert np.all(nzhw == nzhw2)
            imgs = (imgs.astype(np.float32) - 128) / 128
            # bboxes: (z,x,y,diam)
            return torch.from_numpy(imgs), bboxes, torch.from_numpy(coord2), np.array(nzhw)

    def __len__(self):
        if self.phase == 'train':
            return int(len(self.bboxes) / (1 - self.r_rand))
        elif self.phase == 'val':
            return len(self.bboxes)
        else:
            return len(self.sample_bboxes)
