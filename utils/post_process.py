import torch
import torch.nn.functional as F
import numpy as np
class Process:
    def __init__(self,config):
        self.eps = 1e6
        self.max_num_bboxes = config['top_k'] # 100该如何设置？
        self.stride = config['stride']

    def isPeak(self,p,submap):
        submap = submap.ravel()
        cnt = 0
        for confidence in submap:
            if abs(confidence - p) <= self.eps: ++cnt
            if cnt > 1: return False
            if confidence > p: return False
        return True
    def get_centers(self,heatmap):
        nz,nh,nw = heatmap.shape
        centers = []
        for i in range(nz):
            for j in range(nh):
                for k in range(nw):
                    z_left = max(0,i-1)
                    z_right = min(nz,i+2)
                    h_left = max(0,j-1)
                    h_right = min(nh,j+2)
                    w_left = max(0,k-1)
                    w_right = min(nw,k+2)
                    if self.isPeak(heatmap[i,j,k],heatmap[z_left:z_right,h_left:h_right,w_left:w_right]):
                        centers.append(np.array([heatmap[i,j,k],j,k,i,0]))
        idxs = np.argsort(centers[:,0])[:self.max_num_bboxes]
        return centers[idxs]
    def get_pbb(self,output):
        heatmap,offset,diam = output
        bboxes = self.get_centers(heatmap)
        idxs = []
        for i in range(len(bboxes)):
            x,y,z = bboxes[1:-2]
            bboxes[1:-2] = (bboxes[1:-2] + offset[z,x,y]) * self.stride
            if diam[z,x,y] > 0 :
                bboxes[-1] = diam[z,x,y]
                idxs.append(i)
        return bboxes[idxs]


    def _gather_feature(self,feat,ind,mask=None):
        '''
        :param feat: (N,d*h*w,dim)
        :param ind: (N,obj_num,)
        :param mask:
        :return: feat(N,obj_num,dim)
        '''
        # gather feature
        dim = feat.size(-1)
        new_ind = ind.unsqueeze(2).expand(ind.size(0),ind.size(1),dim).long()
        #print("feat size:",feat.size())
        #print("new_ind size:",new_ind.size())
        #print("new_ind element type",type(new_ind[0,0,0]))

        feat = feat.gather(dim=1,index=new_ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat) # 和new_ind形状相同
            feat = feat[mask]
            feat = feat.view(-1, dim)  # 得到
        return feat

    def _tranpose_and_gather_feature(self,feat, ind):
        '''
        :param feat: (N,dim,D,H,W)
        :param ind: (N,obj_num)
        :return: feat(N,obj_num,dim)
        '''
        feat = feat.permute(0,2,3,4,1).contiguous() # （N,D,H,W,dim）
        feat = feat.view(feat.size(0),-1,feat.size(-1)) # （N,D*H*W,dim）
        feat = self._gather_feature(feat,ind)
        return feat

    def objpred_decode(self,hmap,offset,diam):
        pass


def _nms(hmap,kernel_size = 3):
    '''
    maxpool input size与output size相同，固定stride=1，则需要计算padding = (kernel_size -1) // 2,output_size = (input_size + 2 * padding - kernel_size) / stride + 1
    :param hmap:
    :param kernel_size:
    :return:
    '''
    #print("hmap:",hmap)
    hmap = hmap.unsqueeze(dim=0).unsqueeze(dim=0)
    peak = F.max_pool3d(hmap, kernel_size = kernel_size,stride = 1, padding = (kernel_size-1) // 2)
    print("peak:",peak)
    mask = (hmap == peak).float()
    print(mask.size())
    return (hmap * mask).squeeze(dim=0).squeeze(dim=0)

def get_pbb(output, stride=4,threshold = -3):
    #depth,height,width = output.size(0),output.size(1),output.size(2)
    hmap = output[:,:,:,0]
    offset = output[:,:,:,1:4]
    diam = output[:,:,:,4]
    _hmap = torch.from_numpy(hmap)
    mask = (_nms(_hmap).int() > threshold).int()
    inds = torch.nonzero(mask)
    preds = []
    for ind in inds:
        z, x, y = ind
        ofz = offset[z][x][y][0]
        ofx = offset[z][x][y][1]
        ofy = offset[z][x][y][2]
        oz = (z + ofz) * stride
        ox = (x + ofx) * stride
        oy = (y + ofy) * stride
        d = diam[z][x][y]
        prob = hmap[z][x][y]
        preds.append([prob,oz.item(), ox.item(), oy.item(), d])
    print("preds:",preds)
    return preds






import collections
def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]


if __name__ == '__main__':
    data = torch.rand((2,2,2,1))
    print(data)
    out = F.max_pool3d(data,kernel_size = 3, padding = 1, stride = 1)
    print(out)