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
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from utils.util import setgpu, Logger

'''
you can modify $ and run command to test model for each epoch between [start_epoch,epochs]
you can also the parameters in argsparse default value
the results files including pbb.npy and lbb.py will be saved in the path save-dir + "val" + $ for each epoch
nohup python infer.py  --save-dir $ --start-epoch $  --epochs $  > infer.out &

for example:
    nohup python infer.py --save-dir 'results/res18_old/config0'   --start-epoch 100  --epochs 120  --gpu '0' --n_test 2 > infer.out &

'''

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')

parser.add_argument('--model', '-m', metavar='MODEL', default='dpn3d26',
                    help='model')

parser.add_argument('--config', '-c', default='config_training0', type=str)

parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of dataset loading workers (default: 32)')

parser.add_argument('--epoch', default=150, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('--save-dir', default='EXP_RES/dpn3d26', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')

parser.add_argument('--resume', default='234.ckpt', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--testthresh', default=-3, type=float,
                    help='threshod for get pbb')

parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')

parser.add_argument('--gpu', default='1', type=str, metavar='N',
                    help='use gpu')

parser.add_argument('--n_test', default=2, type=int, metavar='N',
                    help='number of gpu for test')

parser.add_argument('--bdir', default='/home/ren/zyx', type=str)

parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 16)')

def transpose(feature):
    n,c,d,h,w = feature.size()
    out = feature.view(n,c,-1)
    out = out.transpose(1,2).view(n,d,h,w,c)
    return out

def test(epoch,data_loader, net, get_pbb, save_dir, config):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'val' + epoch)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    with torch.no_grad():
        for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
            # data: 一个病人的所有splits (nzhw[0]*nzhw[1]*nzhw[2],side_len,side_len,side_len)
            # target: 一个病人的所有GT结节bbox (N,1,1,1,1) -> (id,x,y,z,diam)
            target = [np.asarray(t, np.float32) for t in target]
            lbb = target[0]
            nzhw = nzhw[0]
            name = data_loader.dataset.filenames[i_name].split('/')[-1].split('_clean')[
                0]  # .split('-')[0]
            data = data[0][0]
            coord = coord[0][0]
            isfeat = False
            if 'output_feature' in config:
                if config['output_feature']:
                    isfeat = True
            n_per_run = args.n_test # 每一个测试样本包含几个splits 模型输入数据形状(n_per_run,128,128,128)
            splitlist = list(range(0, len(data) + 1, n_per_run))
            if splitlist[-1] != len(data):
                splitlist.append(len(data))
            outputlist = []
            featurelist = []

            for i in range(len(splitlist) - 1):
                input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
                inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
                if isfeat:
                    output, feature = net(input, inputcoord)
                    feature = transpose(output)
                    featurelist.append(feature.data.cpu().numpy())
                else:
                    output = net(input, inputcoord)
                output = transpose(output)
                outputlist.append(output.data.cpu().numpy())

            output = np.concatenate(outputlist, 0)
            output = split_comber.combine(output, nzhw=nzhw) # 将模型输出combine
            if isfeat:
                #feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
                #feature = split_comber.combine(feature, sidelen)[..., 0]
                feature = split_comber.combine(feature,side_len)
            thresh = args.testthresh  # -8 #-3
            print('pbb thresh', thresh)
            pbb, mask = get_pbb(output, thresh, ismask=True)
            if isfeat:
                feature_selected = feature[mask[0], mask[1], mask[2]]
                np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
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
    print('start!')
    args = parser.parse_args()
    data_base_dir = args.bdir
    config_training = import_module("config." + args.config)
    config_training = config_training.config
    torch.manual_seed(0)

    model = import_module("model." + args.model)
    config, net, loss, get_pbb = model.get_model()
    #start_epoch = args.start_epoch
    #max_epoch = args.epochs
    save_dir = args.save_dir
    print("model:", args.model)
    print("save_dir:", save_dir)
    #print("max_epoch:", args.epochs)
    print("batch_size:", args.batch_size)

    if not save_dir:
        exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
        save_dir = os.path.join("results", args.model + '-' + exp_id)
    else:
        save_dir = os.path.join(save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    test_logfile = os.path.join(save_dir, 'test_log')
    sys.stdout = Logger(test_logfile)

    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu

    # loss = loss.cuda()

    testdatadir = config_training['preprocess_result_path'] #os.path.join(data_base_dir, config_training['preprocess_result_path'])
    #testfilelist = np.load('./luna_test.npy')
    testfilelist = np.load('./config/data/luna_test.npy')

    print('--------test-------------')
    print('len(testfilelist)', len(testfilelist))

    print('batch_size', args.batch_size)
    margin = 16
    sidelen = 128
    from utils import data
    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
    dataset = data.DataBowl3Detector(
        testdatadir,
        testfilelist,
        config,
        phase='test',
        split_comber=split_comber)
    test_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=data.collate,
        pin_memory=False)
    for i, (data, target, coord, nzhw) in enumerate(test_loader):  # check dataset consistency
        if i >= len(testfilelist) // args.batch_size:
            break
    resume_dir = save_dir#args.resume
    print('resume dir:', resume_dir)
    epoch = args.resume[-8:-5]
    config, net, loss, get_pbb = model.get_model()
    resume = resume_dir + "/detector_{}.ckpt" .format(epoch)
    print('resume:', resume)
    checkpoint = torch.load(resume)
    net.load_state_dict(checkpoint['state_dict'])
    device = 'cuda'
    net = net.to(device)
    cudnn.benchmark = True  # False
    net = DataParallel(net).cuda()
    test(epoch, test_loader, net, get_pbb, save_dir, config)


def test_view(tensor):
    n,c,d,h,w = tensor.size()
    tensor = tensor.view(n,c,-1)
    tensor = tensor.transpose(1,2).view(n,d,h,w,c)
    return tensor

def test_view2(tensor):
    tensor = tensor.permute(0,2,3,4,1)
    return tensor

if __name__ == '__main__':
    #main()
    # nohup python detector_infer.py --gpu 1 & #> detector_infer.txt&
    tensor = torch.rand((2,2,3,3,1))
    tensor1 = transpose(tensor)
    tensor2 = test_view2(tensor)
    print(tensor1)
    print(tensor2)