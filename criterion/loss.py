import numpy as np

import torch
from torch import nn

## anchor based
def hard_mining(neg_output, neg_labels, num_hard):
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels


class Loss(nn.Module):
    def __init__(self, num_hard=0):
        super(Loss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = nn.BCELoss()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train=True):
        batch_size = labels.size(0)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        # classify_loss = self.classify_loss(
        #   torch.cat((pos_prob, neg_prob), 0),
        #  torch.cat((pos_labels[:, 0], neg_labels + 1), 0))
        if len(pos_output) > 0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(
                pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
                neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0, 0, 0, 0]
            classify_loss = 0.5 * self.classify_loss(
                neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0, 0, 0, 0]
        classify_loss_data = classify_loss.item()

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]


# Focal Loss
class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        print("FOCAL LOSS", gamma, alpha)

    def forward(self, input, target):
        target = target.float()
        if input.dim() == 1:
            input = input.unsqueeze(1)
        if target.dim() == 1:
            target = target.unsqueeze(1)

        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        # target = target.view(-1,1)
        target = target.float()
        pt = input * target + (1 - input) * (1 - target)
        logpt = pt.log()
        at = (1 - self.alpha) * target + (self.alpha) * (1 - target)
        logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class FocalLoss(nn.Module):
    def __init__(self, num_hard=0):
        super(FocalLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.classify_loss = BinaryFocalLoss(gamma=2, alpha=0.5, size_average=False)
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

    def forward(self, output, labels, train=True):
        batch_size = labels.size(0)

        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_idcs = labels[:, 0] > 0.5

        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        if len(pos_output) > 0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]
            classify_loss = self.classify_loss.forward(
                pos_prob, pos_labels[:, 0]) + self.classify_loss.forward(
                neg_prob, neg_labels + 1)
            classify_loss = classify_loss / (len(pos_prob) + len(neg_prob))
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0, 0, 0, 0]
            classify_loss = self.classify_loss.forward(
                neg_prob, neg_labels + 1)
            classify_loss = classify_loss / len(neg_prob)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0, 0, 0, 0]
        classify_loss_data = classify_loss.item()

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]
