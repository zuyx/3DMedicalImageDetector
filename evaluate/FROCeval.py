import numpy as np
from noduleCADEvaluationLUNA16 import noduleCADEvaluation
import os
import csv
from multiprocessing import Pool
import functools
import sys

sys.path.append("..")
from layers.bbox import nms as nms


def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def convertcsv(p_name, pbb_path, detp):
    resolution = np.array([1, 1, 1])

    origin = np.load(preprocess_path + p_name[:-8] + '_origin.npy', mmap_mode='r')
    spacing = np.load(preprocess_path + p_name[:-8] + '_spacing.npy', mmap_mode='r')
    extendbox = np.load(preprocess_path + p_name[:-8] + '_extendbox.npy', mmap_mode='r')

    pbb = np.load(pbb_path + p_name, mmap_mode='r')
    diam = pbb[:, -1]

    check = sigmoid(pbb[:, 0]) > detp  # detp为Positive阈值
    pbbold = np.array(pbb[check])
    pbbold = np.array(pbbold[pbbold[:, -1] > 3])  # add new 9 15 diam 大于3，除去直径小于3mm的框

    if isNms == 1:
        pbb = nms(pbbold, nmsthresh)  #
    else:
        pbb = pbbold

    pbb = np.array(pbbold[:, :-1])
    print("prob:", sigmoid(pbb[:, 0]))
    print("prob2:",1 / (1 + np.exp(-pbb[:, 0])))
    pbb[:, 1:] = np.array(pbb[:, 1:] + np.expand_dims(extendbox[:, 0], 1).T)
    pbb[:, 1:] = np.array(pbb[:, 1:] * np.expand_dims(resolution, 1).T / np.expand_dims(spacing, 1).T)

    pos = VoxelToWorldCoord(pbb[:, 1:], origin, spacing)
    rowlist = []

    for nk in range(pos.shape[0]):  # pos[nk, 2], pos[nk, 1], pos[nk, 0]
        rowlist.append([p_name[:-8], pos[nk, 2], pos[nk, 1], pos[nk, 0], diam[nk], 1 / (1 + np.exp(-pbb[nk, 0]))])
    # 写入 pid、位置信息、直径、以及归一化的概率
    # 返回一个病人的所有预测结节框
    return rowlist


def getfrocvalue(results_filename, outputdir):
    '''
    :param results_filename: 存放模型预测每一个病人的结节框
    :param outputdir: froc输出值的路径
    :return:
    '''
    return noduleCADEvaluation(annotations_filename, annotations_excluded_filename, seriesuids_filename,
                               results_filename, outputdir)


def getcsv(predanno_outdir, pbb_path, detp):
    '''
    :param infer_outdir: save the best epoch predanno csv file , 通过dept以及nms之后得到的最终PBB的csv保存路径，不过保存的是世界坐标系，在可视化的时候需要将其转化为图像坐标系
    :param pbb_path:  save the pbb.npy which is generated from infer.py
    :param detp:
    :return:
    '''

    if not os.path.exists(predanno_outdir):
        os.makedirs(predanno_outdir)

    for detpthresh in detp:
        print('detp', detpthresh)
        f = open(predanno_outdir + 'predanno' + str(detpthresh) + '.csv', 'w')
        fwriter = csv.writer(f)
        fwriter.writerow(firstline)
        fnamelist = []
        for fname in os.listdir(pbb_path):
            if fname.endswith('_pbb.npy'):
                fnamelist.append(fname)
                # print fname
                # for row in convertcsv(fname, bboxpath, k):
                # fwriter.writerow(row)
        # # return
        print(len(fnamelist))
        predannolist = p.map(functools.partial(convertcsv, pbb_path=pbb_path, detp=detpthresh), fnamelist)
        # print len(predannolist), len(predannolist[0])
        # covertcsv 返回的数据写入predanno-thresh.csv 中
        for predanno in predannolist:
            # print predanno
            for row in predanno:
                # print row
                fwriter.writerow(row)
        f.close()


def getfroc(predanno_outdir, froc_outdir, detp):
    predannofnamalist = []
    froc_outputdirlist = []
    for detpthresh in detp:
        predannofnamalist.append(predanno_outdir + 'predanno' + str(detpthresh) + '.csv')
        print("outcsvpath:", predanno_outdir + 'predanno' + str(detpthresh) + '.csv')
        froc_outputpath = froc_outdir + 'predanno' + str(detpthresh) + '/'
        froc_outputdirlist.append(froc_outputpath)

        if not os.path.exists(froc_outputpath):
            os.makedirs(froc_outputpath)
    froclist = []
    for i in range(len(predannofnamalist)):
        # 对于不同的阈值，计算froc
        froclist.append(getfrocvalue(predannofnamalist[i], froc_outputdirlist[i]))

    np.save(froc_outdir + 'froclist.npy', froclist)


'''
def get_test_data_list(folds):
    path = data_dir + 'luna16/'
    for fold in folds:
        subset = 'subset' + str(fold) + '.npy'
'''

if __name__ == '__main__':
    anchor = 0
    isNms = 0
    data_dir = '/home/ren/zyx/datasets/luna16/'
    preprocess_path = '/home/ren/zyx/datasets/luna16/procdata/'

    annotations_filename = data_dir + 'labels/annos.csv'
    annotations_excluded_filename = data_dir + 'labels/excluded.csv'
    seriesuids_filename = data_dir + 'labels/test_luna_ID_fold6.csv'

    nmsthresh = 0.2  # nmsthresh 越大，删去的冗余框就越少；反之，nmthresh越小，删去的冗余框就越大
    detp = [0.9]
    nprocess = 12

    firstline = ['seriesuid', 'coordX', 'coordY', 'coordZ', 'diam', 'probability']

    model = 'resunet'  # 'dpn3d26'#'res18_ca_d'
    best_epoch = 140
    pbb_path = '../EXP_RES/' + model + '/val' + str(best_epoch) + '/'
    print("pbb_path:", pbb_path)

    predanno_outdir = 'results/' + model +'/'
    froc_outdir = predanno_outdir

    p = Pool(nprocess)
    getcsv(predanno_outdir, pbb_path, detp)
    getfroc(predanno_outdir, froc_outdir, detp)
    p.close()
    print('finished!')

