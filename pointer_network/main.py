"""
list-sort using Pointer Networks
https://arxiv.org/pdf/1506.03134.pdf

Author. Nell (dongboxiang.nell@gmail.com)
Repos. https://github.com/TeNell 

"""

import random
import torch
from torch import Tensor
from torch import nn, optim
from torch.nn import functional as F, Parameter
from torch.autograd import Variable
import numpy as np
import model_train
import model_test
import utils

class PTR:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_net = model_train.PtrNet(self.batch_size, 1, 512, 1, 0)
        self.test_net = model_test.PtrNet(self.batch_size, 1, 512, 1, 0)
        self.latest_model = None
        self.optimizer = torch.optim.Adam(self.train_net.parameters(), lr=1e-4)

    def gen_batch(self):
        batch_size = self.batch_size
        min_len = 5
        max_len = 20
        
        SEQ = np.zeros((batch_size, max_len + 1), dtype='float32')
        IDX = np.zeros((batch_size, max_len + 1), dtype='int64')
        TAG = np.zeros((batch_size, max_len + 1), dtype='float32')
    
        SEQ_m = np.zeros((batch_size), dtype='int64')

        for i in range(batch_size):
            x = np.array(random.sample(range(1,100),
                                       random.randint(min_len, max_len)))
            y = np.argsort(x) + 1
        
            SEQ[i,1:len(x)+1] = x
            SEQ_m[i] = len(x) + 1
            TAG[i,1:len(x)+1] = x[y-1]
            IDX[i,:len(y)] = y
    
        DB = [SEQ, SEQ_m, TAG, IDX]
        return DB 
    
    def train(self):
        DB = self.gen_batch()
        seq = Variable(Tensor(DB[0]))
        seq_m = Variable(torch.LongTensor(DB[1].astype('int64')))
        target = Variable(Tensor(DB[2]))
        label = Variable(torch.LongTensor(DB[3].astype('int64')))
        
        pointer = self.train_net(seq, seq_m, target)
        loss = utils.compute_loss(pointer, label, target)
        acc = utils.compute_acc(pointer, label)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return pointer, loss, acc, label
        
    def test(self):
        self.test_net.load_state_dict(self.train_net.state_dict())
        
        DB = self.gen_batch()
        seq = Variable(Tensor(DB[0]))
        seq_m = Variable(torch.LongTensor(DB[1].astype('int64')))
        target = Variable(Tensor(DB[2]))
        label = Variable(torch.LongTensor(DB[3].astype('int64')))

        pointer = self.test_net(seq, seq_m, target)
        loss = utils.compute_loss(pointer, label, target)
        acc = utils.compute_acc(pointer, label)
        return pointer, loss, acc, label
        
        
        
###################### loop ########################

current_time = utils.get_current_time()    
ptr = PTR(128)

step = 0
while True:

    pointer, loss, acc, label = ptr.train()

    if step % 10 == 0:
        print('step:{0:<6d}loss:{1:<1.11f}acc:{2:<1.11f}'.format(
               step, loss.data.numpy()[0], acc))
    if step % 100 == 0:
        pred = pointer.max(2)[1].squeeze().data.numpy()
        lb = label.data.numpy()
        for i in range(3):
            print('pred:',pred[i,:])
            print('true:',lb[i,:])              
        
    if step % 1000 == 0:
        modelname = 'modelparam_save/ptrnet_para_sort_step-{}.pkl'.format(step)
        torch.save(ptr.train_net.state_dict(), modelname)
        print(modelname, '  Saved!')
        ptr.latest_model = modelname
        
        iter_num, loss_all, acc_all = 0, 0, 0
               
        for _ in range(10):
            pointer, loss, acc, label = ptr.test()
            iter_num += 1; loss_all += loss.data.numpy(); acc_all += acc        
            print('test:{0:<6d}loss:{1:<1.11f}acc:{2:<1.11f}'.format(
                   iter_num, loss.data.numpy()[0], acc))
            pred = pointer.max(2)[1].squeeze().data.numpy()
            lb = label.data.numpy()
            for i in range(3):
                print('pred:',pred[i,:])
                print('true:',lb[i,:]) 
                
        loss_mean = loss_all / iter_num
        acc_mean = acc_all / iter_num
        print('test_loss_mean: ', loss_mean[0])
        print('test_acc_mean:  ', acc_mean)
                          
    step += 1





    
    
    

    
