

import numpy as np

'''
    手动解一元三次方程困难，且误差较大
    另外通过一个channel来预测，点的depth，对于点的depth要怎么得到label信息呢？
    只要当预测的heatmap为目标中心点时，才统计depth的损失
    另外还需要一个offset和diameter
    (N,C,H,W,(k+1+3)) ：(k+1+3) 代表n_classes,depth，3代表offset by stride 
    输入模型:（N,C,D,H,W）
    backbone输出形状：(N,C,D//stride,H//stride,W//stride) 
    输出模型：(N,C,H,W,k+1+3) 这样必然减少了D维度的信息，这就不必训练出一个三维的feature map了
'''
'''
输入图片为三维时还是要解一元三次方程
解一元三次方程可以用逼近法，也可以用直接法求
输入:（N,C,D,H,W）
输出:(N,C,D,H,W,(K+1+3)) 检测有无结节时，K=1 即输出(N,C,D,H,W,5)
'''


def gaussian3D(shape, sigma=1):
    '''
    :param shape:
    :param sigma:
    :return:
    '''
    d, m, n = [(ss - 1.) / 2. for ss in shape]
    z, y, x = np.ogrid[-d:d + 1, -m:m + 1, -n:n + 1]
    # y = [[[-m],[-m+1],[-m+2],...,[1],[2],...,[m]]] # (1,2*m+1,1)
    # x = [[[-m,-m+1,-m+2,...,1,2,...,m]]] #(1,1,2*m+1)
    # z = [[[-m]],[[-m+1]],[[-m+2]],...,[m]] #(2*m+1,1,1)
    # np.array 数组相乘 * ，是标量相乘，对应元素相乘
    h = np.exp(-(x * x + y * y + z * z) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0  # 获得h.eps，把h数组中小于h.eps * h.max()的数设置为0
    return h


def gaussian_radius3d(det_size, min_overlap=0.68):
    diam = det_size  # 直径
    iou_3 = pow(min_overlap, 1 / 3)
    # print("iou**1/3:",iou_3)
    r1 = (diam / iou_3 - diam) / 2
    # print("r1:",r1)
    r2 = (1 - iou_3) * diam / 2
    # print("r2:",r2)
    tmp = pow(min_overlap * 2 / (min_overlap + 1), 1 / 3)
    # print("tmp:",tmp)
    r3 = (1 - tmp) * diam
    # print("r3:",r3)
    # 为什么取min呢？
    # debug
    return 1#min(r1, r2, r3)


def test_radius():
    # bbox = np.random.random((4,)) # (z,y,x,diam)
    bbox = np.array([113., 223., 156., 18.])
    # print("bbox:",bbox)
    radius = gaussian_radius3d(det_size=bbox[3])
    print("radius:", radius)
    z, y, x = bbox[:3]
    # 4个方向中取1个
    new_bbox = np.array([z + radius, y + radius, x + radius, bbox[3]])
    left = bbox[:3] - bbox[3] / 2
    right = bbox[:3] + bbox[3] / 2

    left2 = new_bbox[:3] - new_bbox[3] / 2
    right2 = new_bbox[:3] + new_bbox[3] / 2

    overlap = []
    for i in range(3):
        overlap.append(max(0, min(right[i], right2[i]) - max(left[i], left2[i])))
    intersection = overlap[0] * overlap[1] * overlap[2]
    union = bbox[3] * bbox[3] * bbox[3] + new_bbox[3] * new_bbox[3] * new_bbox[3] - intersection
    # print("uion:",union)
    iou = intersection / union
    print(iou)


def draw_umich_gaussian3d(heatmap, center, radius, k=1):
    '''
    :param heatmap: np.array() shape:(D//stride,H//stride,W//stride)
    :param center: one bbox 的中心
    :param radius: CornerNet中绿色框也可以是GT，(left,right) + radius, (b,u) + radius 确定的框与GT框的IOU值大于等于0.7，radius是满足这个条件的最大半径
    :param k:
    :return:
    '''
    diameter = 2 * radius + 1
    z, y, x = int(center[0]), int(center[1]), int(center[2])
    if radius == 0:
        heatmap[z, y, x] = 1.
        return heatmap
    gaussian = gaussian3D((diameter, diameter, diameter), sigma=diameter / 6)  #

    depth, height, width = heatmap.shape[0:3]
    radius = int(radius)

    up, down = min(z, radius), min(depth - z, radius + 1)

    left, right = min(x, radius), min(width - x, radius + 1)  # 防止center_x - radius < 0 ，center_x + radius > width
    top, bottom = min(y, radius), min(height - y, radius + 1)  # 防止center_y - radius < 0 , center_y + radius > height

    masked_heatmap = heatmap[z - up:z + down, y - top:y + bottom, x - left:x + right]  # heatmap shape: (h,w)

    masked_gaussian = gaussian[radius - up:radius + down, radius - top:radius + bottom,
                      radius - left:radius + right]  # gaussian_masked的中心是(radius,radius),其围成的方框与上面计算的匹配

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)  # 将计算出的gaussian计入heatmap中
    return heatmap


def paint_heatmap(heatmap):
    '''
    :param heatmap: (h,w)
    :return:
    '''
    pass


import torch
debug = False

class LabelMapping(object):
    def __init__(self, config, phase):
        self.stride = np.array(config['stride'])
        self.n_classes = 1
        self.phase = phase
        self.gaussian_iou = 0.7
        self.fmap_size = [size // self.stride for size in config['crop_size']]  # (d,h,w)
        self.max_objs = max(self.fmap_size)
        self.crop_size = config['crop_size']

    def __call__(self, target, bboxes, labels):
        '''
        #:param input_size: (d,h,w)
        :param target: (x,y,z,d)
        :param bboxes: [(x,y,z,d),(x,y,z,d)...]
        :param filename:
        :return:
        虽然当前处理的剪裁图像是围绕着target剪裁的，
        但是在非target区域可能也包含着bboxes里面的结节，
        因为他们都是从一个病人的三维CT图像中采样得到的。
        因此对于(24,24,24)标签，可以mapping bboxes里的结节
        '''
        depth, height, width = self.fmap_size[0], self.fmap_size[1], self.fmap_size[2]

        heatmap = np.zeros([self.n_classes] + self.fmap_size)
        offset = np.zeros((self.max_objs, 3), dtype=np.float32)
        diameters = np.zeros((self.max_objs, 1), dtype=np.float32)
        inds = np.zeros((self.max_objs,), dtype=np.float32)
        ind_masks = np.zeros((self.max_objs,), dtype=np.float32)
        #cnt = 0
        #valid_point_z = []
        #valid_point_diam = []
        #print("target bbox:", target)
        #print("crop size:", self.crop_size)
        if debug:
            valid_bbox = []
        for k, (bbox, label) in enumerate(zip(bboxes, labels)):
            # 要考虑bbox不在cropped_image里的情况
            #print("bbox {} {}".format(k, bbox))
            if any(bbox < 0) or bbox[3] == 0:  # 如果bbox不在cropped_image里内，则不考虑
                continue
            center_int = bbox[:3] // self.stride
            if center_int[0] >= self.fmap_size[0] or center_int[1] >= self.fmap_size[1] or center_int[2] >= \
                    self.fmap_size[2]:
                # 不在feature map 里的center point 不考虑
                continue
            radius = 2
            '''
            #det_diam = bbox[3] // self.stride
            #radius = max(0, gaussian_radius3d(det_diam, self.gaussian_iou))
            '''
            #cnt += 1
            #valid_point_z.append(center_int[0])
            #valid_point_diam.append(radius)
            if debug:
                valid_bbox.append(bbox)
            # heatmap只有label对应的那一张切片会修改heatmap
            draw_umich_gaussian3d(heatmap[label], center_int, radius)
            offset[k] = bbox[:3] / self.stride - center_int  # 相对于fmap
            diameters[k] = bbox[3]
            # offset 和 diameter的loss平衡
            # fmap:(D,H,W)
            # bbox(z,y,x,diam)
            z, y, x = center_int
            inds[k] = z * width * height + y * width + x
            ind_masks[k] = 1

        if debug:
            valid_bbox = np.array(valid_bbox)
            return {
                'hmap': torch.from_numpy(heatmap),
                'offset': torch.from_numpy(offset),
                'diam': torch.from_numpy(diameters),
                'inds': torch.from_numpy(inds),
                'ind_masks': torch.from_numpy(ind_masks),
                'target': torch.from_numpy(valid_bbox)
            }
        else:
            return {
                'hmap': torch.from_numpy(heatmap),
                'offset': torch.from_numpy(offset),
                'diam': torch.from_numpy(diameters),
                'inds': torch.from_numpy(inds),
                'ind_masks': torch.from_numpy(ind_masks),
            }




if __name__ == '__main__':
    print("labelmapping")