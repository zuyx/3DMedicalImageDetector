import torch
import torch.nn.functional as F
import torch.nn as nn

# anchor free
eps = 1e-6

def entropy_loss(preds,targets):
    pos_inds = targets > 1 - eps
    neg_inds = targets < 1 - eps
    preds = F.sigmoid(neg_inds)
    correct_pos_inds = preds[pos_inds] >= 0.5
    correct_neg_inds = preds[neg_inds] < 0.5
    preds = preds.view(preds.size(0),preds.size(1),-1)
    targets = targets.view(targets.size(0),targets.size(1),-1)
    sum_loss = 0.0
    for i in range(preds.size(0)):
        loss = 0.0
        for j in range(preds.size(1)):
            loss += nn.BCELoss(preds[i],targets[i],reduction='sum')
        loss = loss / preds.size(1)
        sum_loss += loss
    loss = sum_loss / preds.size(0)
    return loss, pos_inds.size(),correct_pos_inds.size(),neg_inds.size(),correct_neg_inds.size()

# 对于heatmap而言，0是相对更多的，而1是相对更少的
# 因此为了平衡正负样本带来的loss贡献，对负样本乘以一个权重，来降低负样本的损失贡献
# 同时，focal loss 还引入了一个难样本概念
# 比如对于正样本而言，如果预测的概率是0.9，那么这个样本是正样本的概率很高，那么就可以相对减少这样的样本的损失。
# 对于负样本而言，如果预测的概率是0.1，那么这个样本是负样本的概率很高，属于简单样本，即能够简单预测出的样本。同样可以减少这样样本的损失贡献。
# 并且focal loss 是通过动态地调整共享因子
# 对于 y = 1 : (1-pred_y)^gamma
# 对于 y = 0 : pred_y ^ gamma
# 负样本的权重，仍然是根据位置以及预测的概率调整。



def lesion_cls_loss(preds,targets,gamma = 2):
    '''
    :param preds: (N,1,D,H,W)
    :param targets: (N,1,D,H,W)
    :return:
    '''

    n,c,_,_,_ = targets.size()

    targets = targets.permute(1,0,2,3,4).view(c,n,-1)
    preds = preds.permute(1,0,2,3,4).view(c,n,-1)

    pos_inds = targets.eq(1).float() # 掩码
    neg_inds = targets.lt(1).float() # 掩码

    neg_weights = torch.pow(1 - targets, 4)

    preds = torch.clamp(torch.sigmoid(preds), min = 1e-4, max = 1 - 1e-4)
    pos_preds = (pos_inds * preds)
    neg_preds = (neg_inds * preds)
    pos_loss = (torch.log(pos_preds) * torch.pow(1 - pos_preds ,gamma)).sum(dim=-1)
    neg_loss = (torch.log(1 - neg_preds) * torch.pow(neg_preds,gamma) * neg_weights).sum(dim=-1)

    pos_num = pos_inds.sum(dim=-1) #(c,n)
    neg_num = neg_inds.sum(dim=-1) #(c,n)
    #print("pos_num",pos_num.size())
    #print("neg_num",neg_num.size())
    #print("pos_loss:",pos_loss.size())
    #print("neg_loss:",neg_loss.size())
    if pos_num > 0:
        loss = pos_loss / pos_num + neg_loss / neg_num
    else:
        loss = neg_loss / neg_num
    loss = loss.mean()
    #correct_pos_num = (pos_preds.detach() >= 0.5).float().view(c,-1).sum(dim=1)
    #correct_neg_num = (neg_preds.detach() < 0.5).float().view(c,-1).sum(dim=1)
    #pos_num = pos_num.detach().sum(dim=-1) # (c,1)
    #neg_num = neg_num.detach().sum(dim=-1) # (c,1)
    return loss#, correct_pos_num/(pos_num + 1e-4), correct_neg_num / (neg_num + 1e-4), pos_num, neg_num

def binary_focal_loss(preds,targets,prob = 0.9):
    '''
    :param preds: （N，C,D,H,W)
    :param targets: (N,C,D,H,W)
    :return:
    '''
    n,_,_,_,_ = preds.size()
    #print(preds.size())
    #print(targets.size())

    preds = preds.squeeze(dim=1).view(n,-1)
    targets = targets.squeeze(dim=1).view(n,-1)
    #print(preds.size())
    #print(targets.size())
    pos_inds = targets.eq(1).float() #targets > prob + eps
    neg_inds = targets.lt(1).float()#targets < prob - eps
    #print(pos_inds.size())
    #print(neg_inds.size())
    preds = torch.clamp(torch.sigmoid(preds), min=1e-4, max=1 - 1e-4)  # 限制preds的元素在(min,max)范围内

    #pos_prob = preds[pos_inds]
    #neg_prob = preds[neg_inds]
    #print("pos_prob:",pos_prob)
    #print("neg_prob:",neg_prob)
    #correct_pos_num = (pos_prob > 0.5 + eps).float().sum()
    #correct_neg_num = (neg_prob < 0.5 - eps).float().sum()

    neg_weights = torch.pow(1 - targets, 4)
    loss = 0.0
    for i,pred in enumerate(preds):
        #print(pred.size())
        #print(pos_inds.size())
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds[i]
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights[i] * neg_inds[i]
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        pos_num = pos_inds[i].float().sum()
        neg_num = neg_inds[i].float().sum()
        if pos_num > 0:
            loss -= (pos_loss / pos_num + neg_loss / neg_num)
        else:
            loss -= neg_loss / neg_num
    #pos_num = pos_inds.float().sum()
    #neg_num = neg_inds.float().sum()
    return loss#,correct_pos_num / (pos_num + eps),pos_num,correct_neg_num / (neg_num + eps),neg_num

def get_tpr_tnr(preds,targets,thred = 0.95):
    preds = preds.detach()
    targets = targets.detach()
    pos_inds = targets.eq(1)
    neg_inds = targets.lt(1)
    correct_pos_num = (preds[pos_inds] > 0.5).float().sum()
    correct_neg_num = (preds[neg_inds] < 0.5).float().sum()
    pos_num = pos_inds.float().sum()
    neg_num = neg_inds.float().sum()
    tpr = 0.0
    tnr = 0.0
    if pos_num > 0:
        tpr = correct_pos_num / pos_num
    if neg_num > 0:
        tnr = correct_neg_num / neg_num
    return tpr,tnr,pos_num,neg_num


def _cls_loss(preds,targets):
    pos_inds = targets.eq(1).float() # 只有一个点，而有的时候，甚至没有点，也就是都为0
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4) #周围的position 为变得很小很小，因此这些位置的neg loss 并不会占很大比重。

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4) # 限制pred的元素在(min,max)范围内
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss / len(preds)


def _reg_loss(regs, gt_regs, mask):
    '''
    :param regs: (N,max_objs,dim)
    :param gt_regs:(N,max_objs,dim)
    :param mask: (N,max_objs)
    :return:
    '''
    # 不应该只有mask的那部分才计算loss吗?
    #print("regs:",regs)
    #print("gt_regs:",gt_regs)
    #print("mask:",mask)
    #print("mask == 1 count",(mask == 1).float().sum())
    mask = mask.unsqueeze(2).expand(mask.size(0),mask.size(1),regs.size(2)).float()
    #print("regs size:", (regs * mask).size())
    #print("gt_regs size:", gt_regs.size())
    #print("mask size:",mask.size())
    loss = F.smooth_l1_loss(regs * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) # F.l1_loss(input,target,reduction) input与target的形状大小必须相同，且最后计算时都会转化成(N,1),(N,1)，最后得到的是一个标量
    return loss

import numpy as np
if __name__ == '__main__':
    np.random.seed(2)
    arr = np.random.random((2,8,2))
    tensor = torch.tensor(arr)
    targets = np.random.random((2,8,2))
    targets = torch.tensor(targets)
    masks = np.random.random((2,8))
    masks = torch.tensor(masks)
    masks = torch.where(masks > 0.5,1,0)
    print(tensor)
    print(targets)
    print(masks)
    loss = _reg_loss(tensor,targets,masks)
