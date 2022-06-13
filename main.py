
import argparse

parser = argparse.ArgumentParser(description='PyTorch NoduleDetector')
parser.add_argument('--model', '-m', metavar='MODEL', default='resunet',
                    help='model')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of dataset loading workers (default: 32)')
parser.add_argument('--epochs', default=250, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default='140', type=int, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--save-dir', default='EXP_RES/resunet', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
parser.add_argument('--gpu', default='all', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--loss',default='',type = str)
parser.add_argument('--anchor',default='0',type=int,help='anchor free or anchor based method to detect objects')

from trainer import AnchorfreeTrainer,AnchorBasedTrainer
from dataset import exp_config
from torch.utils.data import DataLoader
from dataset.data import data_config

import numpy as np
import time
import os
import torch
from torch.nn import DataParallel
from model import model_interface
from utils.util import Logger
import sys

def anchorfree_prep():
    from dataset.anchor_free.data import DetectorDataset as Dataset
    train_dataset = Dataset(data_dir, train_patient_ids, config, phase='train')
    val_dataset = Dataset(data_dir, val_patient_ids, config, phase='val')

    model = model_interface.get_anchorfree_model(args.model,config)

    return train_dataset,val_dataset,model

def anchorbased_prep():
    from dataset.anchor_based.data import DataBowl3Detector

    train_dataset = DataBowl3Detector(data_dir, train_patient_ids, config, phase='train', split_comber=None)
    val_dataset = DataBowl3Detector(data_dir, val_patient_ids, config, phase='val')

    model = model_interface.get_anchorbased_model(args.model,config)

    return train_dataset, val_dataset, model

if __name__ == '__main__':
    global args
    args = parser.parse_args()

    from utils.util import setgpu
    device_ids,device = setgpu(args.gpu)
    #device = torch.device('cpu')
    np.random.seed(317)
    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

    prep_config = data_config.config

    config = exp_config.get_config()
    if args.model == 'scpm':
        config['stride'] = 2
    else:
        config['stride'] = 4

    data_dir = prep_config['preprocess_result_path']
    train_patient_ids = np.load(prep_config['train_patient_ids'])
    val_patient_ids = np.load(prep_config['test_patient_ids'])

    if not args.save_dir:
        exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        save_dir = os.path.join("EXP_RES", args.model + '-' + exp_id)
    else:
        save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("batch_size:", args.batch_size)
    print("train_data path:", prep_config['train_patient_ids'])
    print("test_data path:", prep_config['test_patient_ids'])
    print("save_dir:",save_dir)
    print("train patient num:",len(train_patient_ids))
    print("val patient num:",len(val_patient_ids))


    logfile = os.path.join(save_dir,'train_log.out')
    sys.stdout = Logger(logfile)

    if args.anchor == 1:
        train_dataset, val_dataset, model = anchorbased_prep()
    else:
        train_dataset, val_dataset, model = anchorfree_prep()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    model = DataParallel(model,device_ids=device_ids).to(device)
    #model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(),args.lr,momentum=0.9,weight_decay=args.weight_decay)

    if args.anchor == 1:
        from criterion.loss import FocalLoss
        loss = FocalLoss(config['num_hard'])
        trainer = AnchorBasedTrainer(
            epochs=args.epochs,
            lr = args.lr,
            loss=loss,
            device=device,
            checkpoint_path=save_dir,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )
    else:
        trainer = AnchorfreeTrainer(
            epochs=args.epochs,
            lr = args.lr,
            config=config,
            device=device,
            checkpoint_path=save_dir,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader
        )

    if args.resume:
        resume = save_dir + '/detector_%03d.ckpt' % args.resume
        print("resume:",resume)
        trainer.load_checkpoint(resume=resume,loadOptimizer=True)

    print("start training!")

    #debug:
    #trainer.validate()
    trainer.train()



