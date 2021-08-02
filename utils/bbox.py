import torch
from torch import nn
import numpy as np

class GetPBB(object):
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output, thresh=-3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx, yy, zz, aa = np.where(mask)

        output = output[xx, yy, zz, aa]
        if ismask:
            return output, [xx, yy, zz, aa]
        else:
            return output

        # output = output[output[:, 0] >= self.conf_th]
        # bboxes = nms(output, self.nms_th)




nms_type = 'soft_Linear' # 'soft_Gaus' 'hard'




def soft_nms(bboxes,nms_thresh = 0.3, soft_thresh = 0.3,sigma = 0.5):
    '''
    :param bboxes:
    :param nms_thresh:
    :param soft_thresh:
    :return:
    '''
    bboxes = bboxes[np.argsort(-bboxes[:,0])]
    weights = bboxes[:,0]
    keep_box = []
    while len(order)!=0:
        top_box = bboxes[order[0]]
        keep_box.append(top_box)
        if len(order)==1:
            break
        ious = [ iou(top_box,box)  for box in  bboxes[1:]]
        idx = np.squeeze(np.nonzero(ious >=nms_thresh))
        weights[idx] *= np.exp(-(ious[idx] * ious[idx])/sigma)

        idx = weights[idx] >= soft_thresh
        if len(idx) == 0: # 当前top bbox 将剩余的box都删去了
            break
        bboxes = bboxes[idx]
        bboxes[:,0] = weights[idx]
        order = np.argsort(-bboxes[:,0])

    return keep_box
'''
def box_soft_nms(bboxes, nms_threshold=0.3, soft_threshold=0.3, sigma=0.5, mode='union'):
    """
    两个thresh，一个nms_thresh，来筛选出，与top-scored IOU值超过nms_thresh的框，这些框的score是需要rescore的，具体可用线性函数、高斯核、sigmoid等函数rescore
    一个soft_thresh用来将rescore后的score值太小的box删去
    soft-nms implentation according the soft-nms paper
    :param bboxes: all pred bbox
    :param nms_threshold: origin nms thres, for judging to reduce the cls score of high IoU pred bbox
    :param soft_threshold: after cls score of high IoU pred bbox been reduced, soft_thres further filtering low score pred bbox
    :return:
    """

    box_keep = []
    weights = bboxes[:,0].copy()
    x1 = bboxes[:, 1]
    y1 = bboxes[:, 2]
    z1 = bboxes[:, 3]
    d = bboxes[:, 4]
    #areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # bbox面积
    order = np.argsort(-weights[:])
    while len(order) > 0:  # 对应NMS中step-5
        i = order[0]  # 当前order中的top-1，保存之
        box_keep.append(c_boxes[i])  # 保存bbox

        if len(order) == 1:  # 当前order就这么一个bbox了，那不玩了，下一个类的bbox操作吧
            break


        # 经过origin NMS thres，得到高IoU的bboxes index，
        # origin NMS操作就直接剔除掉这些bbox了，soft-NMS就是对这些bbox对应的score做权重降低
        ids_t = (ovr >= nms_threshold).nonzero().squeeze()  # 高IoU的bbox，与inds = np.where(ovr >= nms_threshold)[0]功能类似

        weights[[order[ids_t + 1]]] *= torch.exp(-(ovr[ids_t] * ovr[ids_t]) / sigma)

        # soft-nms对高IoU pred bbox的score调整了一次，soft_threshold仅用于对score抑制，score太小就不考虑了
        ids = (weights[order[1:]] >= soft_threshold).nonzero().squeeze()  # 这一轮未被抑制的bbox
        if len(ids) == 0:  # 竟然全被干掉了，下一个类的bbox操作吧
            break

        c_boxes = bboxes[order[1:]][ids]  # 先取得c_boxes[order[1:]]，再在其基础之上操作[ids]，获得这一轮未被抑制的bbox
        c_boxes[:,0] = weights[order[1:]][ids]
        _, order = np.argsort(-c_boxes[:,0]) #c_scores.sort(0, descending=True)
        if c_boxes.dim() == 1:
            c_boxes = c_boxes.unsqueeze(0)

        x1 = c_boxes[:, 1]  # 因为bbox已经做了筛选了，areas需要重新计算一遍，抑制的bbox剔除掉
        y1 = c_boxes[:, 2]
        x2 = c_boxes[:, 3]
        y2 = c_boxes[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #print("box_kepp",type(box_keep))
    print("box_keep len:",len(box_keep))
    return np.asarray(box_keep, np.float32)#, scores_keep

'''

def nms(output, nms_th):
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])] # 根据置信度从大到小排列
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th: # 和top-scored box 的IOU值大于nms_th ，因其与top-scored的重叠度不在允许的范围内，所以将其删除
                flag = -1
                break
        if flag == 1: #
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes

'''
hard nms后处理的方法比较粗暴，
直接将与top-scored box的IOU值超过给定阈值的box删除，
即设置其weight为0。这会导致结果受nmsthresh影响较大。
试想，如果两个bbox框，bbox1的置信度大于bbox2的置信度，但这个置信度却很靠近时，而真实结节其实更靠近bbox2的时候，这时就会提高FP数
def nms(output,nms_th):
    if len(output) == 0:
        return output
    output = output(np.argsort(-output[:,0])) # argsort(-list)返回[第一大元素的下标，第二大元素的下标，...，最小元素的小标]
'''



def iou(box0, box1):
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


def acc(pbb, lbb, conf_th, nms_th, detect_th):
    pbb = pbb[pbb[:, 0] >= conf_th]
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score > bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p, [bestscore]], 0))
            else:
                fp.append(np.concatenate([p, [bestscore]], 0))
        if flag == 0:
            fp.append(np.concatenate([p, [bestscore]], 0))
    for i, l in enumerate(lbb):
        if l_flag[i] == 0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5], l))
            if len(score) != 0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore < detect_th:
                fn.append(np.concatenate([l, [bestscore]], 0))

    return tp, fp, fn, len(lbb)


def topkpbb(pbb, lbb, nms_th, detect_th, topk=30):
    conf_th = 0
    fp = []
    tp = []
    while len(tp) + len(fp) < topk:
        conf_th = conf_th - 0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th < -3:
            break
    tp = np.array(tp).reshape([len(tp), 6])
    fp = np.array(fp).reshape([len(fp), 6])
    fn = np.array(fn).reshape([len(fn), 5])
    allp = np.concatenate([tp, fp], 0)
    sorting = np.argsort(allp[:, 0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk, len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
    #     print(fp_in_topk)
    fn_i = np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp = allp[:topk]
    if len(fn_i) > 0:
        fn = np.concatenate([fn, tp[fn_i, :5]])
    else:
        fn = fn
    if len(tp_in_topk) > 0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk) > 0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp, fn

