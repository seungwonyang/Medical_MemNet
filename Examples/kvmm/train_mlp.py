#!/usr/bin/python
from __future__ import division
from Model import MLP
from config import config
from utils import *
import pdb
import torch
import torch.autograd as autograd
import torch.nn as nn
# from torch.legacy.nn import CosineEmbeddingCriterion
# from torch.nn import CosineEmbeddingCriterion
import torch.nn.functional as F
import torch.optim as optim

import os
import socket
import math
import random
import torchvision

class Partition(object):

   def __init__(self, data_q, data_a, data_sim, index):
       self.data_q = data_q
       self.data_a = data_a
       self.data_sim = data_sim
       self.index = index

   # def __len__(self):
   #     return len(self.index)

   # def __getitem__(self, index):
   #     data_idx = self.index[index]
   #     print 'data index', data_idx
   #     print 'data size', len(self.data_q[data_idx])
   #     return self.data_q[data_idx], self.data_a[data_idx], self.data_sim[data_idx]

   def extract_data(self, index):
       data_idx = self.index[index]
       print 'data index', data_idx
       print 'data size', len(self.data_q[data_idx])
       return self.data_q[data_idx], self.data_a[data_idx], self.data_sim[data_idx]


class DataPartitioner(object):

   def __init__(self, data_q, data_a, data_sim, sizes=[0.7, 0.2, 0.1], seed=1234):
       self.data_q = data_q
       self.data_a = data_a
       self.data_sim = data_sim

       self.partitions = []
       rng = random.Random()
       rng.seed(seed)
       data_len = len(data_q)
       print 'data size', data_len
       indexes = [x for x in range(0, data_len)]
       rng.shuffle(indexes)

       for frac in sizes:
           part_len = int(frac * data_len)
           print part_len
           self.partitions.append(indexes[0:part_len])
           indexes = indexes[part_len:]

   def use(self, partition):
       indexes = self.partitions[partition]
       # print 'indexes', indexes
       # return Partition(self.data_q, self.data_a, self.data_sim, self.partitions[partition])
       part_data_q = [self.data_q[x] for x in indexes]
       part_data_a = [self.data_a[x] for x in indexes]
       part_data_sim = [self.data_sim[x] for x in indexes]

       return part_data_q, part_data_a, part_data_sim

def partition_dataset():
   train_q, train_a, train_sim = load_from_file("./pkl/sim/train_pair.pkl")
   # train_q, train_a, train_sim = load_from_file("./pkl/sim/dev_pair.pkl")
   print 'data loaded...'
   # dev_q, dev_a, dev_sim = load_from_file("./pkl/sim/dev_pair.pkl")
   # train_q_iter = batch_sort_iter(train_q, config.batch_size, config.epoch, padding = True)
   # train_a_iter = batch_sort_iter(train_a, config.batch_size, config.epoch, padding = True, sort=False)
   # train_sim_iter = batch_sort_iter(train_sim, config.batch_size, config.epoch, padding = False)
   rank_num = 0 # dist.get_rank()
   world_size = 10000 # torch.distributed.get_worlhd_size()
   data_size = len(train_q)
   dist_size = int(data_size/world_size) + 1
   print 'data size', data_size
   print 'dist size', dist_size
   start_index = rank_num * dist_size
   end_index = min((rank_num + 1) * dist_size, data_size)

   return train_q[start_index:end_index], train_a[start_index:end_index], train_sim[start_index:end_index]

   # bsz = int(config.batch_size/size)
   # partition_sizes = [1.0 / size for _ in range(size)]

   # partition = DataPartitioner(train_q, train_a, train_sim, partition_sizes)
   # partition = partition.use(0) # partition.use(torch.distributed.get_rank())
   # return partition

# train_q, train_a, train_sim = load_from_file("./pkl/sim/train_pair.pkl")
# train_q, train_a, train_sim = load_from_file("./pkl/sim/dev_pair.pkl")

# print 'train size', len(train_q)
# train_q_iter = batch_sort_iter(train_q, config.batch_size, config.epoch, padding = True)
# train_a_iter = batch_sort_iter(train_a, config.batch_size, config.epoch, padding = True, sort=False)
# train_sim_iter = batch_sort_iter(train_sim, config.batch_size, config.epoch, padding = False)

# large dev evaluation will result in memory issue
# N = 1000
# dev_q_t = to_tensor(dev_q[:N], padding = True) 
# dev_a_t = to_tensor(dev_a[:N], padding = True, sort=False)
# dev_sim_t = torch.LongTensor(dev_sim[:N])

model = MLP(config)
optimizer = optim.SGD(model.parameters(), lr=config.lr)
#optimizer = nn.StochasticGradient(mlp, criterion)
#criterion = CosineEmbeddingCriterion(config.margin)

#pdb.set_trace()

def train(): 
    cnt = 0
    dataset = partition_dataset()
    print 'size', len(dataset[0])
    train_q_iter = batch_sort_iter(dataset[0], config.batch_size, config.epoch, padding = True)
    train_a_iter = batch_sort_iter(dataset[1], config.batch_size, config.epoch, padding = True, sort=False)
    train_sim_iter = batch_sort_iter(dataset[2], config.batch_size, config.epoch, padding = False)

    for i_q, i_a, i_s in zip(train_q_iter, train_a_iter, train_sim_iter):
        #pdb.set_trace()
        model.zero_grad()
        loss = model.forward(i_q, i_a, i_s.type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        print "Training loss", loss.data.sum()
        # cnt += 1
        # if cnt % config.valid_every == 0:
        #     loss = model.forward(dev_q_t, dev_a_t, dev_sim_t)
        #     print "Validation loss", loss.data.sum()
train()
model.save(config.pre_embed_file)
