import numpy as np
import random

# 对每一个patch采样的voxel生成3个anchor，对这些anchor分配与之IOU最大的真实目标框
class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.num_neg = int(config['num_neg'])  # 最大neg数量，dpn3d26中设置为800
        self.th_neg = config['th_neg']
        self.anchors = np.asarray(config['anchors'])
        self.phase = phase
        if phase == 'train':
            self.th_pos = config['th_pos_train']
        elif phase == 'val':
            self.th_pos = config['th_pos_val']

    def __call__(self, input_size, target, bboxes, filename):
        '''
        :param input_size:
        :param target:
        :param bboxes:
        :param filename:
        :return:
        虽然当前处理的剪裁图像是围绕着target剪裁的，
        但是在非target区域可能也包含着bboxes里面的结节，
        因为他们都是从一个病人的三维CT图像中采样得到的。
        因此对于(24,24,24,3,5)标签，可以mapping bboxes里的结节
        '''
        stride = self.stride
        num_neg = self.num_neg
        th_neg = self.th_neg
        anchors = self.anchors
        th_pos = self.th_pos

        output_size = []
        for i in range(3):
            if input_size[i] % stride != 0:
                # input_size无法整除stride
                print("input_size[i]%stride!=0",filename)
            # assert(input_size[i] % stride == 0)
            # 每一个方向上有多少个anchor中心点
            output_size.append(input_size[i] // stride)  # 整除
        # print("----test len anchors:",len(anchors))
        # print("test outputsize + anchors,",output_size + [len(anchors), 5])
        # 列表之间的加号是拼接
        # label得到的是形状为(output_size[0],output_size[1],output_size[2],len(anchors),5)的元素全为1.0的矩阵
        label = -1 * np.ones(output_size + [len(anchors), 5], np.float32)
        offset = ((stride.astype('float')) - 1) / 2
        # offset的作用是？
        # 对每个维度，每隔stride长度，采样一个点，作为目标框的中心点
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)  #
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        for bbox in bboxes:  # 遍历一个病人的所有真实结节目标框
            for i, anchor in enumerate(anchors):  # 对所有采样点遍历每个尺度的生成目标框
                # 每个采样点生成len(anchors)个尺度的目标框
                iz, ih, iw = select_samples(bbox, anchor, th_neg, oz, oh, ow)
                # 给定bbox和anchor如果二者的IOU值超过th_neg == 0.02则设置标签为0
                label[iz, ih, iw, i, 0] = 0  # 只要有一个bbox与位置(z,x,y)及第i个anchor的IOU值超过0.02，那么就设置标签为0
        # print("label size:",label.size())

        if self.phase == 'train' and self.num_neg > 0:
            neg_z, neg_h, neg_w, neg_a = np.where(label[:, :, :, :, 0] == -1)  # 负例坐标
            neg_idcs = random.sample(range(len(neg_z)), min(num_neg, len(neg_z)))  #
            # 随机采样min(num_neg,len(neg_z))个下标，这一步的目的是让负例不超过800
            neg_z, neg_h, neg_w, neg_a = neg_z[neg_idcs], neg_h[neg_idcs], neg_w[neg_idcs], neg_a[neg_idcs]
            label[:, :, :, :, 0] = 0
            # 为什么负例label设为-1？
            label[neg_z, neg_h, neg_w, neg_a, 0] = -1



        if np.isnan(target[0]):  # 如果这一剪裁的三维图像数据中本就不包含真实目标框，则直接返回label
            return label
        iz, ih, iw, ia = [], [], [], []
        # 如果预测到的目标框都是负例，那么很有可能，生成的框都是负例？
        for i, anchor in enumerate(anchors):
            # 返回与target的iou大于0.5的anchor的中心位置，标记为正例
            iiz, iih, iiw = select_samples(target, anchor, th_pos, oz, oh, ow)
            iz.append(iiz)  # [1,2,3],[7,8],[3,4,5]
            ih.append(iih)
            iw.append(iiw)
            ia.append(i * np.ones((len(iiz),), np.int64))  # ia是什么？
        iz = np.concatenate(iz, 0)  # 列方向合并
        ih = np.concatenate(ih, 0)
        iw = np.concatenate(iw, 0)
        ia = np.concatenate(ia, 0)
        flag = True
        if len(iz) == 0:  # 正例个数为0,重新生成anchor
            pos = []
            for i in range(3):
                pos.append(max(0, int(np.round((target[i] - offset) / stride))))
            idx = np.argmin(np.abs(np.log(target[3] / anchors)))  # ?
            pos.append(idx)
            flag = False
        else:
            # 从range(0,1,2,3,..mlen(iz)-1)中选取一个数作为target对应的anchor的下标
            # 为什么不选IOU最大的那个？
            idx = random.sample(range(len(iz)), 1)[0]
            pos = [iz[idx], ih[idx], iw[idx], ia[idx]]  #
        dz = (target[0] - oz[pos[0]]) / anchors[pos[3]]
        dh = (target[1] - oh[pos[1]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dw = (target[2] - ow[pos[2]]) / anchors[pos[3]]
        dd = np.log(target[3] / anchors[pos[3]])
        label[pos[0], pos[1], pos[2], pos[3], :] = [1, dz, dh, dw, dd]
        return label
        # 返回的是一个[24,24,24,3,5]的标签,对于IOU在0.02到0.5之间的位置，其标签设置为0，标签为0代表着可能包含着非target的结节，而标签为-1代表着一定不含结节。
        # 对于与所有bboxes的IOU都小于0.02的位置，其标签设置为-1
        # 对于与target的IOU超过0.5的位置，随机选取一个作为正例，标签设置为1，为什么是随机选取？不应该选最大的那个吗？
        # 注意这里只对一个位置(z,y,x),一个anchor大小，设置正标签
        # 也就是每一个围绕target裁剪的图像(96,96,96)，只有一个target，且只有一个正例标签(1),其余的anchor的标签都设置为-1或0
        # 问题来了，对于FPN，正例个数会不会太少？->看懂loss部分


def select_samples(bbox, anchor, th, oz, oh, ow):
    z, h, w, d = bbox
    max_overlap = min(d, anchor)

    # print(np.power(max(d, anchor), 3) * th,max_overlap,max_overlap)
    min_overlap = np.power(max(d, anchor), 3) * th / max_overlap / max_overlap
    # print('bbox',bbox, min_overlap, max_overlap)
    # print(bb)
    if min_overlap > max_overlap:
        return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)
    else:
        s = z - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = z + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mz = np.logical_and(oz >= s, oz <= e)
        iz = np.where(mz)[0]

        s = h - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = h + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mh = np.logical_and(oh >= s, oh <= e)
        ih = np.where(mh)[0]

        s = w - 0.5 * np.abs(d - anchor) - (max_overlap - min_overlap)
        e = w + 0.5 * np.abs(d - anchor) + (max_overlap - min_overlap)
        mw = np.logical_and(ow >= s, ow <= e)
        iw = np.where(mw)[0]

        if len(iz) == 0 or len(ih) == 0 or len(iw) == 0:
            return np.zeros((0,), np.int64), np.zeros((0,), np.int64), np.zeros((0,), np.int64)

        lz, lh, lw = len(iz), len(ih), len(iw)
        iz = iz.reshape((-1, 1, 1))
        ih = ih.reshape((1, -1, 1))
        iw = iw.reshape((1, 1, -1))
        iz = np.tile(iz, (1, lh, lw)).reshape((-1))
        ih = np.tile(ih, (lz, 1, lw)).reshape((-1))
        iw = np.tile(iw, (lz, lh, 1)).reshape((-1))
        centers = np.concatenate([
            oz[iz].reshape((-1, 1)),
            oh[ih].reshape((-1, 1)),
            ow[iw].reshape((-1, 1))], axis=1)

        r0 = anchor / 2
        s0 = centers - r0
        e0 = centers + r0

        r1 = d / 2
        s1 = bbox[:3] - r1
        s1 = s1.reshape((1, -1))
        e1 = bbox[:3] + r1
        e1 = e1.reshape((1, -1))

        overlap = np.maximum(0, np.minimum(e0, e1) - np.maximum(s0, s1))
        # print('overlap',e0,e1,s0,s1)
        intersection = overlap[:, 0] * overlap[:, 1] * overlap[:, 2]
        union = anchor * anchor * anchor + d * d * d - intersection

        # print(intersection, union)
        iou = intersection / union

        mask = iou >= th
        # if th > 0.4:
        #   if np.sum(mask) == 0:
        #      print(['iou not large', iou.max()])
        # else:
        #    print(['iou large', iou[mask]])
        iz = iz[mask]
        ih = ih[mask]
        iw = iw[mask]
        return iz, ih, iw