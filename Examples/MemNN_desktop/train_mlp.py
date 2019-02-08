### #!/usr/bin/python
from __future__ import division
from Model import MLP
from config import config
from utils import *
import pdb
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.legacy.nn import CosineEmbeddingCriterion
import torch.nn.functional as F
import torch.optim as optim
import time

train_q, train_a, train_sim = load_from_file("./pkl/sim/train_pair.pkl")
dev_q, dev_a, dev_sim = load_from_file("./pkl/sim/dev_pair.pkl")

num_batches_per_epoch = int(len(train_q)/config.batch_size) + 1
total_batch_size = num_batches_per_epoch * config.epoch
print "{} batches will be processed".format(total_batch_size)

train_q_iter = batch_sort_iter(train_q, config.batch_size, config.epoch, padding = True)
train_a_iter = batch_sort_iter(train_a, config.batch_size, config.epoch, padding = True, sort=False)
train_sim_iter = batch_sort_iter(train_sim, config.batch_size, config.epoch, padding = False)

# large dev evaluation will result in memory issue
N = 1000
dev_q_t = to_tensor(dev_q[:N], padding = True) 
dev_a_t = to_tensor(dev_a[:N], padding = True, sort=False)
dev_sim_t = torch.FloatTensor(dev_sim[:N])

model = MLP(config)
optimizer = optim.SGD(model.parameters(), lr=config.lr)
#optimizer = nn.StochasticGradient(mlp, criterion)
#criterion = CosineEmbeddingCriterion(config.margin)

#pdb.set_trace()
# 5618800 batches expected
# print type(train_q_iter)
# a_iter = iter(train_a_iter)
# print a_iter.next()

# print sum(1 for x in train_q_iter)
# print sum(1 for x in train_a_iter)
# print sum(1 for x in train_sim_iter)

def train(): 
    cnt = 0
    a_iter = iter(train_a_iter)
    q_iter = iter(train_q_iter)
    s_iter = iter(train_sim_iter)

    # huge memory is consumed about 168GB - 1701GB when using zip function
    # for i_q, i_a, i_s in zip(train_q_iter, train_a_iter, train_sim_iter):
    for i in range(total_batch_size):
        #pdb.set_trace()
        start = time.time()
        model.zero_grad()
        # loss = model.forward(i_q, i_a, i_s)
        loss = model.forward(q_iter.next(), a_iter.next(), s_iter.next().type(torch.FloatTensor))
        loss.backward()
        optimizer.step()
        print i, "Training loss", loss.data.sum()
        if loss.data.sum() < 0.011:
            break
        cnt += 1
        if cnt % config.valid_every == 0:
            loss = model.forward(dev_q_t, dev_a_t, dev_sim_t)
            print "Validation loss", loss.data.sum()
        print time.time() - start
    print 'count', cnt
train()
model.save(config.pre_embed_file)
