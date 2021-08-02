from criterion.criterion import _cls_loss, _reg_loss, lesion_cls_loss, binary_focal_loss,get_tpr_tnr
from utils.post_process import Process
import numpy as np
import time
import torch
from torch.autograd import Variable

class BaseTrainer(object):
    def __init__(self, epochs,lr, device=None, checkpoint_path=None, model=None, optimizer=None, train_loader=None,
                 val_loader=None):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.iterations = 1
        self.max_epochs = epochs
        self.bestLoss = 1000
        self.save_dir = checkpoint_path
        self.lr = lr

    def get_lr(self):
        if self.iterations <= self.max_epochs * 1 / 3:  # 0.5:
            lr = self.lr
        elif self.iterations <= self.max_epochs * 2 / 3:  # 0.8:
            lr = 0.1 * self.lr
        elif self.iterations <= self.max_epochs * 0.8:
            lr = 0.05 * self.lr
        else:
            lr = 0.01 * self.lr
        return lr

    def save(self):
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        checkpoint_path = self.save_dir + 'detector_%03d.ckpt' % self.iterations
        torch.save(
            {'epoch': self.iterations + 1,
             'state_dict': state_dict,
             'best_loss': self.bestLoss,
             'optimizer': self.optimizer.state_dict(),
             },
            checkpoint_path
        )
        print("save model on epoch %d" % self.iterations)

    def load_checkpoint(self, resume_dir, loadOptimizer):
        if resume_dir is not None:
            print("loading checkpoint...")
            model_dict = self.model.state_dict()
            modelCheckpoint = torch.load(resume_dir)
            pretrained_dict = modelCheckpoint['state_dict']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            # 更新model_dict中包含的参数
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
            self.model.load_state_dict(model_dict)
            print("loaded model finished!")
            self.iterations = modelCheckpoint['epoch']
            if 'best_loss' in modelCheckpoint.keys():
                self.bestLoss = modelCheckpoint['best_loss']
            if loadOptimizer == True:
                self.optimizer.load_state_dict(modelCheckpoint['optimizer'])
                print('loaded! optimizer')
            else:
                print('not loaded optimizer')
        else:
            print('No checkpoint is included')
    def epoch_train(self):
        pass
    def validate(self):
        pass
    def train(self):
        start_epoch = self.iterations
        for epoch in range(start_epoch, self.max_epochs + 1):
            self.epoch_train()
            valiloss = self.validate()
            if self.bestLoss >= valiloss:
                self.bestLoss = valiloss
                self.save()
            self.iterations += 1
    def debug(self):
        print("--------------------start debug for parameters----------------")
        params = []
        for param in self.model.parameters():
            params.append(param)

        #grads = [x.grad for x in self.optimizer.param_groups[0]['params']]
        #for grad in grads:
        #    print("grad:", grad)
        return params

    def debug_after_backwords(self,params):

        params_step = []

        for param in self.model.parameters():
            params_step.append(param)
        for i in range(len(params)):
            print("参数更新前后差别：",(params_step[i] == params[i]).float().sum())

    def single_test(self):
        pass

class AnchorfreeTrainer(BaseTrainer):

    def __init__(self,epochs,lr,config,device = None,checkpoint_path = None,model=None, optimizer=None, train_loader=None,val_loader = None):
        super(AnchorfreeTrainer, self).__init__(
            epochs, lr,device, checkpoint_path, model, optimizer, train_loader, val_loader
        )
        self.cls_criterion = lesion_cls_loss
        loss_name = 0
        if loss_name == 1:
            self.cls_criterion = _cls_loss
            print("using _cls_loss!")
        else:
            self.cls_criterion = binary_focal_loss#lesion_cls_loss
            print("using binary_focal_loss!")
        self.get_tpr_tnr = get_tpr_tnr
        #self.cls_criterion = binary_focal_loss
        self.reg_criterion = _reg_loss
        self.pprocess = Process(config=config)
        self.print_fq = 5

    def epoch_train(self):
        self.model.train()
        start_time = time.time()

        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        losses = 0.0
        hmap_losses = 0.0
        diam_losses = 0.0
        offset_losses = 0.0

        TPR = 0.0
        TNR = 0.0
        posNum = 0
        negNum = 0
        batch_num = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            for k in batch:
                #print(k)
                batch[k] = Variable(batch[k].to(device=self.device, non_blocking=True),requires_grad = True)
                #print(batch[k].requires_grad)

            #print("batch['image]:",batch['image'])
            fmap,hmap,offset,diam = self.model(batch['image'],batch['coord'])
            #print("diam output:",diam)
            offset = self.pprocess._tranpose_and_gather_feature( feat = offset, ind = batch['inds'])
            diam = self.pprocess._tranpose_and_gather_feature(feat = diam, ind = batch['inds'])

            #print("hmap:",hmap)
            #print("target hmap:",batch['hmap'])
            #print("diam transform:",diam)
            #print("taret hmap==1:",(batch['hmap'] - 1 < 1e6).float().sum())

            hmap_loss = self.cls_criterion(hmap,batch['hmap'])
            tpr,tnr,pos_num,neg_num = self.get_tpr_tnr(hmap,batch['hmap'])
            #hmap_loss, tpr, pos_num, tnr ,neg_num = self.cls_criterion(hmap, batch['hmap'])
            offset_loss = self.reg_criterion(offset, batch['offset'], batch['ind_masks'])
            diam_loss = self.reg_criterion(diam, batch['diam'], batch['ind_masks'])

            #print("hmap loss:", hmap_loss)
            #print("diam_loss:",diam_loss)
            #print("offset_loss:",offset_loss)

            loss = hmap_loss + 1 * offset_loss + 0.1 * diam_loss
            #loss = hmap_loss
            #loss = diam_loss

            #params = self.debug()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            #self.debug_after_backwords(params)

            losses += loss.item() # item()得到的是元素值
            hmap_losses += hmap_loss.item()
            diam_losses += diam_loss.item()
            offset_losses += offset_loss.item()
            batch_num += 1.0

            # metrics
            # TP/ P = TP / (TP + FN)
            # TN / N = TN / (TN + FP)
            # only valid for one classes data
            if batch_idx % self.print_fq == 0:
                print("Train  epoch:{} batch_idx:{}  tpr : {} , tnr: {}, pos_num: {}, neg_num: {}".format(
                    self.iterations, batch_idx, tpr, tnr, pos_num, neg_num
                ))

            TPR += tpr
            TNR += tnr
            posNum += pos_num
            negNum += neg_num

        end_time = time.time()
        print('Epoch %03d (lr %.5f)' % (self.iterations, lr))
        print('Train:      tpr %3.2f, tnr %3.2f, total pos points num %d, total neg points num %d, time %3.2f' % (
            100.0 * TPR / batch_num,
            100.0 * TNR / batch_num,
            posNum,
            negNum,
            end_time - start_time))

        print('loss %2.4f, classify loss %2.4f, offset regress loss %2.4f, diam regress loss %2.4f'
              %(
                  losses / batch_num,
                  hmap_losses / batch_num,
                  offset_losses / batch_num,
                  diam_losses / batch_num
              ))
        print()

    def validate(self):
        self.model.eval()
        start_time = time.time()

        losses = 0.0
        hmap_losses = 0.0
        diam_losses = 0.0
        offset_losses = 0.0

        TPR = 0.0
        TNR = 0.0
        posNum = 0
        negNum = 0
        batch_num = 0.0

        with torch.no_grad():
            for batch_idx,batch in enumerate(self.val_loader):
                for k in batch:
                    batch[k] = Variable(batch[k].to(device=self.device, non_blocking=True),requires_grad = False)

                fmap,hmap, offset, diam = self.model(batch['image'],batch['coord'])

                offset = self.pprocess._tranpose_and_gather_feature(offset, batch['inds'])
                diam = self.pprocess._tranpose_and_gather_feature(diam, batch['inds'])
                hmap_loss = self.cls_criterion(hmap, batch['hmap'])
                #hmap_loss,tpr,pos_num,tnr,neg_num = self.cls_criterion(hmap, batch['hmap'])
                tpr, tnr, pos_num, neg_num = self.get_tpr_tnr(hmap, batch['hmap'])

                offset_loss = self.reg_criterion(offset, batch['offset'], batch['ind_masks'])
                diam_loss = self.reg_criterion(diam, batch['diam'], batch['ind_masks'])

                loss = hmap_loss + 1 * offset_loss + 0.1 * diam_loss

                #print("hmap_loss:",hmap_loss)
                #print("diam_loss:", diam_loss)
                #print("offset_loss:", offset_loss)

                losses += loss.item()  # item()得到的是元素值
                hmap_losses += hmap_loss.item()
                diam_losses += diam_loss.item()
                offset_losses += offset_loss.item()
                batch_num += 1.0
                # metrics
                # TP/ P = TP / (TP + FN)
                # TN / N = TN / (TN + FP)
                # only valid for one classes data
                if batch_idx % self.print_fq == 0:
                    print("Validation batch_idx:{} tpr : {} , tnr: {}, pos_num: {}, neg_num: {}".format(
                        batch_idx,tpr, tnr, pos_num, neg_num
                    ))
                TPR += tpr
                TNR += tnr
                posNum += pos_num
                negNum += neg_num

            end_time = time.time()

            print('Validation: tpr %3.2f, tnr %3.8f, total pos point num %d, total neg points num %d, time %3.2f' % (
                100.0 * TPR / batch_num,
                100.0 * TNR / batch_num,
                posNum,
                negNum,
                end_time - start_time))

            print('loss %2.4f, classify loss %2.4f, offset regress loss %2.4f, diam regress loss %2.4f'
                  % (
                      losses / batch_num,
                      hmap_losses / batch_num,
                      offset_losses / batch_num,
                      diam_losses / batch_num
                  ))
            print()
        return losses / batch_num
    def single_test(self):
        with torch.no_grad():
            for idx,batch in enumerate(self.val_loader):
                for k in batch.keys():
                    batch[k] = Variable(batch[k].to(device=self.device, non_blocking=True), requires_grad=False)
                fmap,hmap, offset, diam = self.model(batch['image'], batch['coord'])
                offset = self.pprocess._tranpose_and_gather_feature(offset, batch['inds'])
                diam = self.pprocess._tranpose_and_gather_feature(diam, batch['inds'])
                #tpr, tnr, pos_num, neg_num = self.get_tpr_tnr(hmap, batch['hmap'])
                return fmap,batch['image'],batch['hmap'],hmap,offset,diam


class AnchorBasedTrainer(BaseTrainer):
    def __init__(self,epochs,lr,loss = None,device = None,checkpoint_path = None,model=None, optimizer=None, train_loader=None,val_loader = None):
        super(AnchorBasedTrainer, self).__init__(
            epochs,lr, device, checkpoint_path, model, optimizer, train_loader,
            val_loader
        )
        self.loss = loss

    def epoch_train(self):
        start_time = time.time()
        self.model.train()
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        metrics = []
        for i, (data, target, coord) in enumerate(self.train_loader):
            #print("data:",data)
            data = Variable(data.to(self.device), requires_grad=True)
            target = Variable(target.to(self.device))
            coord = Variable(coord.to(self.device), requires_grad=True)
            fmap,output = self.model(data, coord)
            #print("output:",output)
            loss_output = self.loss(output, target)
            #print("output", output)
            #print("loss_output",loss_output)
            #print("loss:", loss_output)

            #params = self.debug()

            self.optimizer.zero_grad()
            loss_output[0].backward()
            self.optimizer.step()

            #self.debug_after_backwords(params)

            loss_output[0] = loss_output[0].item()
            metrics.append(loss_output)
        end_time = time.time()

        metrics = np.asarray(metrics, dtype=np.float32)
        print('Epoch %03d (lr %.5f)' % (self.iterations, lr))
        print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            end_time - start_time))
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])))
        print()

    def validate(self):
        start_time = time.time()
        self.model.eval()
        metrics = []
        with torch.no_grad():
            for i, (data, target, coord) in enumerate(self.val_loader):

                data = Variable(data.to(self.device))
                target = Variable(target.to(self.device))
                coord = Variable(coord.to(self.device))
                fmap,output = self.model(data, coord)
                loss_output = self.loss(output, target, train=False)
                loss_output[0] = loss_output[0].item()
                metrics.append(loss_output)
        end_time = time.time()

        metrics = np.asarray(metrics, np.float32)
        #print("metrics:",metrics)
        print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            end_time - start_time))
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])))
        print
        print
        return np.mean(metrics[:, 0])

    def single_test(self):
        with torch.no_grad():
            for i, (data, target, coord) in enumerate(self.val_loader):
                data = Variable(data.to(self.device))
                target = Variable(target.to(self.device))
                coord = Variable(coord.to(self.device))
                fmap,output = self.model(data, coord)
                loss_output = self.loss(output, target, train=False)
                loss_output[0] = loss_output[0].item()
                return data,fmap,output,target

if __name__ == '__main__':
    pass

