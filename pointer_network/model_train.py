"""

Author. Nell (dongboxiang.nell@gmail.com)
Repos. https://github.com/TeNell 

"""
import torch
from torch import nn, optim
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
from torch import Tensor
import copy
import numpy as np
from tqdm import tqdm
import functools
import utils

class PtrNet(nn.Module):
    def __init__(self, batch_size, feat_len, hidden_size, num_layers, dropout):
        super().__init__()
        self.batch_size = batch_size
        self.feat_len = feat_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder = nn.LSTM(self.feat_len, self.hidden_size, self.num_layers,
                               batch_first=True, dropout=self.dropout)
        self.decoder = nn.LSTM(self.feat_len, self.hidden_size, self.num_layers,
                               batch_first=True, dropout=self.dropout)
        self.e_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.d_linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v = Parameter(torch.ones(self.hidden_size))
        
    def forward(self, seq, seq_m, target):
        
        seq = seq.unsqueeze(2)
        target = target.unsqueeze(2)
        batch_size = seq.size()[0]
        e_in_len = seq.size()[1]
        d_in_len = target.size()[1]
        #***********************encoder***********************
        # e: encoder outputs
        # e_j: j-th node of encoder outputs
        # e_h: encoder hidden state H
        # e_c: encoder hidden state C
        e_list, e_h_list, e_c_list = [], [], []
        for j in range(e_in_len):
            seq_j = torch.unsqueeze(seq[:,j,:], 1)
            if j == 0:
                e_j, [e_h_j, e_c_j] = self.encoder(seq_j, )
            else:
                e_j, [e_h_j, e_c_j] = self.encoder(seq_j, [e_h_j, e_c_j])
            e_list.append(e_j)
            e_h_list.append(e_h_j)
            e_c_list.append(e_c_j)

        e = torch.squeeze(torch.stack(e_list)).permute(1, 0, 2).contiguous()
        e_h = torch.stack(e_h_list)
        e_c = torch.stack(e_c_list)
         
        e_hs, e_cs = [], []
        # find real end of each encoder sequence before zero-padding
        for p in range(batch_size):
            e_hs.append(e_h[seq_m[p].data[0]-1, 0, p, :])
            e_cs.append(e_c[seq_m[p].data[0]-1, 0, p, :])

        e_hs = torch.unsqueeze(torch.stack(e_hs), 0)
        e_cs = torch.unsqueeze(torch.stack(e_cs), 0)
        e_state = [e_hs, e_cs]
        #*****************************************************
        
        # train mode:

        #***********************decoder***********************
        # d: decoder outputs
        # d_i: i-th node of decoder outputs
        # d_h: decoder hidden state H
        # d_c: decoder hidden state C
        d_state = e_state
    
        d_list = []        
        for i in range(d_in_len):
            target_i = torch.unsqueeze(target[:,i,:], 1)
            d_i, d_state = self.decoder(target_i , d_state)
            d_list.append(d_i) 
    
        d = torch.squeeze(torch.stack(d_list)).permute(1, 0, 2).contiguous() 
        #*****************************************************
            
        #***********************pointer***********************
        # p: pointer(covering zero-padding part)
        # p_i: pointer, produced by i-th decoder_out and all encoder_outs
        dim1 = batch_size
        dim2 = d_in_len
        dim3 = e_in_len
        dim4 = e_hs.size()[2]        
    
        e = self.e_linear(e.view(-1, e.size()[2])).view(e.size())      
        d = self.d_linear(d.view(-1, d.size()[2])).view(d.size())
    
        v0 = self.v.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        v = torch.tanh(
                e.unsqueeze(1).expand(dim1, dim2, dim3, dim4)
                + d.unsqueeze(2).expand(dim1, dim2, dim3, dim4)
                ) * v0.expand(dim1, dim2, dim3, dim4)
        p = v.sum(3).view(*list(v.size())[:-1])
        #***************************************************** 
            
        #*******************pointer(partial)******************
        # p: pointer(without zero-padding part)
        dim1 = batch_size
        dim2 = e_in_len
        dim3 = d_in_len
        p = p.permute(0, 2, 1).contiguous().view(-1, dim3)
        p = p - 1000
            
        # creat numpy array to realize "partial pointer" 
        sq = seq.data
        sq[:, 0, :] = 0.1
        idx = torch.squeeze(torch.nonzero(torch.mean(sq, 2).view(-1)))
        idx = idx.numpy() 
        
        size_1, size_2 = p.size()
        add_np = np.zeros((size_1, size_2))
        add_np[idx] = 1000
        add_np = Variable(torch.from_numpy(add_np).float())
    
        p = (p + add_np).view(dim1, dim2, dim3).permute(0, 2, 1)
        #***************************************************** 
        
        pointer = utils.softmax_3d(p)
        
        
        return pointer

    
    
