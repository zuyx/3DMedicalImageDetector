
import os
import torch

def setgpu(ids):
    if ids == 'all':
        device_ids = list(range(torch.cuda.device_count()))
        ids = map(str,device_ids)
        ids = ','.join(ids)
    else:
        device_ids = ids.split(',')
        device_ids = [int(id) for id in device_ids]
        print('device_ids',device_ids)
        print("ids:",ids)

    print("using gpu id:",ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = ids

    return device_ids,torch.device('cuda')

import sys
class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

if __name__ == '__main__':
    device_ids,device = setgpu('1')
    print(device_ids)
    print(device)




