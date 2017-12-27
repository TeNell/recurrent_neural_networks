"""

Author. Nell (dongboxiang.nell@gmail.com)
Repos. https://github.com/TeNell 

"""
import torch
from torch import Tensor
from torch.nn import functional as F
import time
import datetime
import numpy as np
import itertools

#------------------------------common---------------------------------
def get_current_time():
    """return time string. eg:2017-09-09-123456
    """
    current_time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')
    return current_time

def softmax_3d(logit):
    return F.softmax(logit.contiguous().view(-1, logit.size()[2])).view(*logit.size())

#---------------------------------------------------------------------


#----------------------------ptrnet_sort------------------------------
def compute_loss(pointer, label, target):

    n, t, d = pointer.size()
    p = pointer.contiguous().view(n * t, d)
    l = label.view(n * t, 1)
    v = p.gather(1, l)
    eps = 1e-6
    v = torch.clamp(v, eps, 1 - eps) 
    
    ti = target.data
    ti[:, 0] = 1
    idx_r = torch.nonzero(torch.mean(ti, 1).view(-1))[:,0]

    v = v[idx_r]

    return -torch.log(v).mean()


def compute_acc(pointer, label):    

    p = pointer.max(2)[1].view(-1) 
    l = label.view(-1) 

    idx_r = torch.nonzero(l.data)[:,0]
    l = l[idx_r]
    p = p[idx_r]
    correct = (
        (p == l).type(torch.FloatTensor).sum().data.numpy()[0])
    total = l.size()[0] 
    return correct / total

def compute_absacc(pointer, label):    

    P = pointer.max(2)[1].squeeze()
    L = label
    count = 0
    for i in range(P.size()[0]):
        p = P[i]
        l = L[i]
        idx = torch.nonzero(l.data)[:,0]
        p = p[idx]
        l = l[idx]
        if torch.equal(p.data, l.data) == True:
            count = count + 1
    correct = count / P.size()[0]
    return correct
#---------------------------------------------------------------------

