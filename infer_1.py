import argparse
import os
import time
from importlib import import_module
import sys

sys.path.append('../')
from utils.split_combine import SplitComb
import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from dataset import exp_config
from torch.utils.data import DataLoader
from dataset.data import data_config
from torch.autograd import Variable
import numpy as np
from utils.util import setgpu, Logger
from model import model_interface
from utils.post_process import get_pbb,collate
from dataset.anchor_free.data import DataDetectorTestDataset as Dataset
'''
you can modify $ and run command to test model for each epoch between [start_epoch,epochs]
you can also the parameters in argsparse default value
the results files including pbb.npy and lbb.py will be saved in the path save-dir + "val" + $ for each epoch
nohup python infer.py  --save-dir $ --start-epoch $  --epochs $  > infer.out &

for example:
    nohup python infer.py --save-dir 'results/res18_old/config0'   --start-epoch 100  --epochs 120  --gpu '0' --n_test 2 > infer.out &

'''

margin = 16
sidelen = 128

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')

parser.add_argument('--model', '-m', metavar='MODEL', default='resunet',
                    help='model')

parser.add_argument('--config', '-c', default='config_training0', type=str)

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of dataset loading workers (default: 32)')

parser.add_argument('--save-dir', default='EXP_RES/resunet', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')

parser.add_argument('--resume', default='140.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--threshold', default= 0.9, type=float,
                    help='threshod for get pbb for anchor free model')

parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')

parser.add_argument('--gpu', default='1', type=str, metavar='N',
                    help='use gpu')

parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')

parser.add_argument('--bdir', default='/home/ren/zyx', type=str)

parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 16)')

def transpose(feature):
    n,c,d,h,w = feature.size()
    out = feature.view(n,c,-1)
    out = out.transpose(1,2).view(n,d,h,w,c)
    return out

def output_transpose(hmap,offset,diam):
    hmap = transpose(hmap) # (n,c)
    offset = transpose(offset)
    diam  = transpose(diam)
    return hmap,offset,diam


def output_collect(hmap,offset,diam):
    hmap = transpose(hmap) # (n,d,h,w,c)
    offset = transpose(offset)
    diam  = transpose(diam)
    out = torch.cat((hmap,offset,diam),dim=-1)
    return out


def test(epoch,data_loader, net, get_pbb, save_dir, config):

    save_dir = os.path.join(save_dir, 'val' + epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    n_per_run = args.n_test  # 每一个测试样本包含几个splits 模型输入数据形状(n_per_run,128,128,128)
    thresh = args.threshold # get_pbb中的阈值

    start_time = time.time()

    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber

    with torch.no_grad():
        '''
            one batch image just include one patient ct images
            for one batch, one patient ct images is splited into nzhw[0] * nzhw[1] * nzhw[2] splits
            put the n_per_run splits per model in the model,and combine them together to get the predicted box bounding .
         '''
        for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
            # data: 一个病人的所有splits (nzhw[0]*nzhw[1]*nzhw[2],side_len,side_len,side_len)
            # target: 一个病人的所有GT结节bbox (N,1,1,1,1) -> (id,x,y,z,diam)
            target = [np.asarray(t, np.float32) for t in target]
            lbb = target[0]
            nzhw = nzhw[0]
            name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[0]  # .split('-')[0]
            data = data[0][0]
            coord = coord[0][0]
            splitlist = list(range(0, len(data) + 1, n_per_run)) #

            if splitlist[-1] != len(data):# 处理最后一份split如果不满 n_per_run 的情况
                splitlist.append(len(data))

            hmap_list = []
            offset_list = []
            diam_list = []
            for i in range(len(splitlist) - 1):
                input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
                _, hmap, offset, diam = net(input, inputcoord)
                hmap = torch.sigmoid(hmap)
                hmap,offset,diam = output_transpose(hmap,offset,diam)
                #print("hmap size:",hmap.size())
                hmap_list.append(hmap.data.cpu().numpy())
                #print("len(hmap_list):",len(hmap_list))
                offset_list.append(offset.cpu().numpy())
                diam_list.append(diam.cpu().numpy())
                # 如何将output拼接起来？
            hmap = np.concatenate(hmap_list, 0)
            #hmap = np.expand_dims(hmap,axis=1)
            offset = np.concatenate(offset_list,0)
            #offset = np.expand_dims(offset,axis=1)
            diam = np.concatenate(diam_list,0)
            #diam = np.expand_dims(diam,axis=1)

            hmap = split_comber.combine(hmap, nzhw=nzhw) # 将模型输出combine
            offset = split_comber.combine(offset,nzhw=nzhw)
            diam = split_comber.combine(diam,nzhw=nzhw)

            pbb= get_pbb(hmap,offset,diam,thresh)
            np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
            np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)

    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print()
    print()

def main():
    '''
    对每一个epoch测试，得到val*文件，val*文件中包含pbb.npy，是模型对每一个病人预测的所有包含结节的bounding box
    在测试完之后，执行evaluate.py，评估模型
    :return:
    '''
    global args
    args = parser.parse_args()
    from utils.util import setgpu
    device_ids, device = setgpu(args.gpu)
    np.random.seed(317)
    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

    prep_config = data_config.config

    config = exp_config.get_config()
    if args.model == 'scpm':
        config['stride'] = 2
    else:
        config['stride'] = 4

    save_dir = args.save_dir
    assert save_dir is not None
    logfile = os.path.join(save_dir,'test_log.txt')
    sys.stdout = Logger(logfile)
    print("test_data path:", prep_config['test_patient_ids'])
    print("save_dir:", save_dir)

    data_dir = prep_config['preprocess_result_path']
    test_patient_ids = np.load(prep_config['debug_patient_ids'])#['test_patient_ids'])

    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])

    dataset = Dataset(
        data_dir,
        test_patient_ids ,
        config,
        split_comber=split_comber
    )

    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=False
    )

    for i, (data, target, coord, nzhw) in enumerate(test_loader):  # check dataset consistency
        if i >= len(test_patient_ids) // args.batch_size:
            break

    epoch = args.resume[-8:-5]
    model = model_interface.get_anchorfree_model(args.model, config)
    resume = save_dir + "/detector_{}.ckpt".format(epoch)
    print('resume:', resume)
    load_check_point(model,resume)
    model = model.to(device)
    cudnn.benchmark = True  # False
    model = DataParallel(model).cuda()
    test(epoch, test_loader, model, get_pbb, save_dir, config)

def load_check_point(model,resume,):
    model_dict = model.state_dict()
    modelCheckpoint = torch.load(resume)
    pretrained_dict = modelCheckpoint['state_dict']
    # 过滤操作
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    # 更新model_dict中包含的参数
    model_dict.update(new_dict)
    # 打印出来，更新了多少的参数
    print('Total : {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)
    epoch = modelCheckpoint['epoch']
    print("loaded model finished!")


def debug():
    arr = [[1, 24, 24, 24, 1], [1, 24, 24, 24, 1]]
    arr = np.array(arr)
    arrs = []
    for i in range(4):
        arrs.append(arr)
    arrs = np.concatenate(arrs, axis=0)
    print(arrs.shape)
    print(arrs)


if __name__ == '__main__':
    main()
    # nohup python detector_infer.py --gpu 1 & #> detector_infer.txt&

