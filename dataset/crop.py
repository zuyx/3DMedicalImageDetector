
from scipy.ndimage import zoom
import warnings
import numpy as np
class Crop(object):#imgs, bbox[1:], bboxes,isScale,isRandom
    def __init__(self, config):
        self.crop_size = config['crop_size']#[96, 96, 96]
        self.bound_size = config['bound_size']#12
        self.stride = config['stride']#4
        self.pad_value = config['pad_value']#170
    def __call__(self, imgs, target, bboxes,isScale=False,isRand=False):
        if isScale:
            radiusLim = [8.,120.]
            scaleLim = [0.75,1.25]
            scaleRange = [np.min([np.max([(radiusLim[0]/target[3]),scaleLim[0]]),1]) ,np.max([np.min([(radiusLim[1]/target[3]),scaleLim[1]]),1])]#target代表bbox[1:]代表[74.7895776 249.75681577 267.72357450000004 10.57235285]
            scale = np.random.rand()*(scaleRange[1]-scaleRange[0])+scaleRange[0]#np.random.rand(d0, d1, …, dn)的随机样本位于[0, 1)中
            crop_size = (np.array(self.crop_size).astype('float')/scale).astype('int')#根据实际结节直径大小调整crop_size大小,target[3]小crop_size变大.target[3]大crop_size变小
        else:
            crop_size=self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)#目标结节，需要围绕着这个结节剪裁
        bboxes = np.copy(bboxes)#这个ct含有的所有结节
        start = []
        for i in range(3):
            if not isRand:
                r = target[3] / 2
                s = np.floor(target[i] - r)+ 1 - bound_size
                e = np.ceil (target[i] + r)+ 1 + bound_size - crop_size[i]
            else:#
                s = np.max([imgs.shape[i+1]-crop_size[i]//2,imgs.shape[i+1]//2+bound_size])
                e = np.min([crop_size[i]//2, imgs.shape[i+1]//2-bound_size])
                target = np.array([np.nan,np.nan,np.nan,np.nan])
            if s>e:
                start.append(np.random.randint(e,s))#!
            else:
                start.append(int(target[i])-crop_size[i]//2+np.random.randint(-bound_size//2,bound_size//2))#求取结节的3d立方矩阵最靠近原点的点


        normstart = np.array(start).astype('float32')/np.array(imgs.shape[1:])-0.5#将normstart放到3d块[-0.5:0.5,-0.5:0.5,-0.5:0.5]对应实际[-imgs.shape[1]/2:imgs.shape[1]/2,-imgs.shape[2]/2:imgs.shape[2]/2,-imgs.shape[1]/2:imgs.shape[1]/2,-imgs.shape[3]/2:imgs.shape[3]/2]
        normsize = np.array(crop_size).astype('float32')/np.array(imgs.shape[1:])
        xx,yy,zz = np.meshgrid(np.linspace(normstart[0],normstart[0]+normsize[0],self.crop_size[0]//self.stride),
                           np.linspace(normstart[1],normstart[1]+normsize[1],self.crop_size[1]//self.stride),#np.linspace创建等差数列
                           np.linspace(normstart[2],normstart[2]+normsize[2],self.crop_size[2]//self.stride),indexing ='ij')#可以这么理解，meshgrid函数用两个坐标轴上的点在平面上画网格(3d也是如此)
        #print(11111111)
        coord = np.concatenate([xx[np.newaxis,...], yy[np.newaxis,...],zz[np.newaxis,:]],0).astype('float32')
        #print(coord)
        pad = []
        pad.append([0,0])
        for i in range(3):
            leftpad = max(0,-start[i])
            rightpad = max(0,start[i]+crop_size[i]-imgs.shape[i+1])
            pad.append([leftpad,rightpad])
        crop = imgs[:,#以结节位置为中心来截取
            max(start[0],0):min(start[0] + crop_size[0],imgs.shape[1]),
            max(start[1],0):min(start[1] + crop_size[1],imgs.shape[2]),
            max(start[2],0):min(start[2] + crop_size[2],imgs.shape[3])]
        crop = np.pad(crop,pad,'constant',constant_values =self.pad_value)#越界进行填充
        for i in range(3):
            target[i] = target[i] - start[i]#目标结节位置已经减去目标结节3d立方矩阵最靠近原点的开始点
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j] = bboxes[i][j] - start[j]#同一ct的所有结节位置已经减去该目标结节3d立方矩阵最靠近原点的开始点

        if isScale:
            #print("isScale:",isScale)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop,[1,scale,scale,scale],order=1)
            newpad = self.crop_size[0]-crop.shape[1:][0]
            if newpad<0:
                crop = crop[:,:-newpad,:-newpad,:-newpad]
            elif newpad>0:
                pad2 = [[0,0],[0,newpad],[0,newpad],[0,newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i]*scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j] = bboxes[i][j]*scale
        return crop, target, bboxes, coord#1*96*96*96/可能是[nan,nan,nan,nan]/可能是空/3*24*24*24
