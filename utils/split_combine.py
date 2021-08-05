import torch
import numpy as np
'''=
split_comber = SplitComb(
sidelen = 128, 
config['max_stride'] = 16, 
config['stride'], 
margin = 16, 
config['pad_value'] = 1
)
'''

'''
just for anchor free model output for the present
'''
class SplitComb():
    def __init__(self,side_len,max_stride,stride,margin,pad_value):
        self.side_len = side_len # ?
        self.max_stride = max_stride # ?
        self.stride = stride # ?
        self.margin = margin # ?
        self.pad_value = pad_value
        # print(side_len,max_stride,stride,margin,pad_value)64 16 4 32 170
    def split(self, data, side_len = None, max_stride = None, margin = None):
        if side_len==None:
            side_len = self.side_len
        if max_stride == None:
            max_stride = self.max_stride
        if margin == None:
            margin = self.margin
        
        assert(side_len > margin)
        assert(side_len % max_stride == 0)
        assert(margin % max_stride == 0)

        splits = []
        _, z, h, w = data.shape
        nz = int(np.ceil(float(z) / side_len))
        nh = int(np.ceil(float(h) / side_len))
        nw = int(np.ceil(float(w) / side_len))
        
        nzhw = [nz,nh,nw] # 每个维度有多个splits

        self.nzhw = nzhw
        pad = [ [0, 0],
                [margin, nz * side_len - z + margin],# 上界扩充margin个pixel，下界扩充nz*sidelen - z + margin 个pixel ，即将data的形状扩充成恰好可以被side_len整除且+margin的形状
                [margin, nh * side_len - h + margin],#32,
                [margin, nw * side_len - w + margin]]#32,

        data = np.pad(data, pad, 'edge') # 将data的形状扩充成恰好可以被side_len整除且+margin的形状

        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len # left_z
                    ez = (iz + 1) * side_len + 2 * margin # right_z # (128 + 16*2 = 160)
                    sh = ih * side_len # left_h
                    eh = (ih + 1) * side_len + 2 * margin # right_h
                    sw = iw * side_len # left_w
                    ew = (iw + 1) * side_len + 2 * margin # right_w
                    split = data[np.newaxis, :, sz:ez, sh:eh, sw:ew]
                    splits.append(split)
                    #print("slpit shape:",split.shape)
        splits = np.concatenate(splits, 0)

        return splits,nzhw

    def combine(self, output, nzhw = None, side_len=None, stride=None, margin=None):
        '''
        :param output: 模型的输出，输出形状为(side_len//stride,side_len//stride,side_len//stride,C)
        :param nzhw: 每个维度有多少个side
        :param side_len:
        :param stride:下采样因子
        :param margin:边界
        :return:
        '''
        if side_len==None:
            side_len = self.side_len
        if stride == None:
            stride = self.stride
        if margin == None:
            margin = self.margin
        if nzhw.any()==None:
            nz = self.nz
            nh = self.nh
            nw = self.nw
        else:
            nz,nh,nw = nzhw
        assert(side_len % stride == 0)
        assert(margin % stride == 0)
        # print('side_len, stride, margin',side_len, stride, margin)#160 4 16
        # print('nz, nh, nw',nz, nh, nw)#2 2 2 
        side_len //= stride # 32
        margin //= stride # 4

        splits = []
        for i in range(len(output)):
            splits.append(output[i])
        #print("splits:",splits)
        output = -1000000 * np.ones((
            nz * side_len,
            nh * side_len,
            nw * side_len,
            splits[0].shape[3]), np.float32)
        # print('ONES-output',output.shape, nzhw)

        idx = 0
        for iz in range(nz):
            for ih in range(nh):
                for iw in range(nw):
                    sz = iz * side_len
                    ez = (iz + 1) * side_len
                    sh = ih * side_len
                    eh = (ih + 1) * side_len
                    sw = iw * side_len
                    ew = (iw + 1) * side_len
                    split = splits[idx][margin:margin + side_len, margin:margin + side_len, margin:margin + side_len] # [4:36]
                    #print("split:",split)
                    #print("split shape:",split.shape)
                    output[sz:ez, sh:eh, sw:ew] = split
                    idx += 1
        return output


if __name__ == '__main__':
    '''
    split_comber = SplitComb(
        side_len = 64,
        max_stride = 16,
        stride = 4,
        margin = 16,
        pad_value = 170
    )
    imgs = np.random.rand((1,95,95,95))
    split_comber.split(imgs)
    '''


