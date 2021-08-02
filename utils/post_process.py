import torch
import torch.nn.functional as F
import numpy as np
class Process:
    def __init__(self,config):
        self.eps = 1e6
        self.max_num_bboxes = config['top_k'] # 100该如何设置？
        self.stride = config['stride']
    '''
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
    def get_bbox(self,output):
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
    '''

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
    peak = F.max_pool3d(hmap, kernel = kernel_size,stride = 1, padding = (kernel_size-1) // 2)
    mask = (hmap == peak).float()
    return hmap * mask


def get_pbb(hmap, offset, diam, stride=4,threshold = 0.9 ):
    mask = (_nms(hmap) > threshold).int()  # (hmap > 0.95).int()
    inds = torch.nonzero(mask)
    print(inds)
    preds = []
    for ind in inds:
        n, z, x, y, c = ind
        ofz = offset[n][z][x][y][0]
        ofx = offset[n][z][x][y][1]
        ofy = offset[n][z][x][y][2]
        oz = (z + ofz) * stride
        ox = (x + ofx) * stride
        oy = (y + ofy) * stride
        d = diam[n][z][x][y][c]
        prob = hmap[n][z][x][y][c]
        preds.append([prob,oz, ox, oy, d])
    return preds









if __name__ == '__main__':
    data = [83, 86, 77, 15, 93, 35, 86, 92, 49, 21, 62, 27, 90, 59, 63, 26, 40, 26, 72, 36, 11, 68, 67, 29, 82, 30, 62,
            23, 67, 35, 29, 2, 22, 58, 69, 67, 93, 56, 11, 42, 29, 73, 21, 19, 84, 37, 98, 24, 15, 70, 13, 26, 91, 80,
            56, 73, 62, 70, 96, 81, 5, 25, 84, 27, 36, 5, 46, 29, 13, 57, 24, 95, 82, 45, 14, 67, 34, 64, 43, 50, 87, 8,
            76, 78, 88, 84, 3, 51, 54, 99, 32, 60, 76, 68, 39, 12, 26, 86, 94, 39]
    data.sort()
    print(data)