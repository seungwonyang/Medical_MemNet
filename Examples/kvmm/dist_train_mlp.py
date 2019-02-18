
import socket

from Model import MLP
from config import config
from utils import *
import pdb
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(self, x1, x2, y):
        # v1 = Variable(x1)
        # v2 = Variable(x2)
        # y = Variable(y)
        v1 = self.embed(x1)
        v1 = v1.mean(1).squeeze(1)
        v2 = self.embed(x2)
        v2 = v2.mean(1).squeeze(1)
        loss = self.cosine(v1,v2,y)
        return loss

    def save(self, filename):
        tmp = [x for x in self.parameters()]
        with open(filename, "w") as f:
            torch.save(tmp[0], f) 

    def load(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed.weight = embed_t

def partition_dataset():
   train_q, train_a, train_sim = load_from_file("./pkl/sim/train_pair.pkl")
   # train_q, train_a, train_sim = load_from_file("./pkl/sim/dev_pair.pkl")
   print 'data loaded...'
   # dev_q, dev_a, dev_sim = load_from_file("./pkl/sim/dev_pair.pkl")
   # train_q_iter = batch_sort_iter(train_q, config.batch_size, config.epoch, padding = True)
   # train_a_iter = batch_sort_iter(train_a, config.batch_size, config.epoch, padding = True, sort=False)
   # train_sim_iter = batch_sort_iter(train_sim, config.batch_size, config.epoch, padding = False)
   rank_num = torch.distributed.get_rank()
   world_size = torch.distributed.get_world_size()
   data_size = len(train_q)
   dist_size = int(data_size/world_size) + 1
   print 'data size', data_size
   print 'dist size', dist_size
   start_index = rank_num * dist_size
   end_index = min((rank_num + 1) * dist_size, data_size)

   return train_q[start_index:end_index], train_a[start_index:end_index], train_sim[start_index:end_index]

def average_gradients(model):
   size = float(torch.distributed.get_world_size())
   for param in model.parameters():
       torch.distributed.all_reduce(param.grad.data, op=torch.distributed.reduce_op.SUM)
       param.grad.data /= size


def run(rank, size):
   hostname = socket.gethostname()
   torch.manual_seed(1234)
   dataset = partition_dataset()
   print 'size', len(dataset[0])
   batch_size = config.batch_size * 64 # 8192
   train_q_iter = batch_sort_iter(dataset[0], batch_size, config.epoch, padding = True)
   train_a_iter = batch_sort_iter(dataset[1], batch_size, config.epoch, padding = True, sort=False)
   train_sim_iter = batch_sort_iter(dataset[2], batch_size, config.epoch, padding = False)
   
   #model = Net()
   gpu_rank = rank % 4
   print('gpu_rank = ', gpu_rank, ' rank = ', rank)
   # model = Net().cuda(gpu_rank)
   model = MLP(config).cuda(gpu_rank)
   # optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.5)
   optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)

   for i_q, i_a, i_s in zip(train_q_iter, train_a_iter, train_sim_iter):
      #pdb.set_trace()
      i_q, i_a, i_s  = torch.autograd.Variable(i_q.cuda(gpu_rank)), torch.autograd.Variable(i_a.cuda(gpu_rank)), torch.autograd.Variable(i_s.type(torch.FloatTensor).cuda(gpu_rank))
      model.zero_grad()
      # loss = model.forward(i_q, i_a, i_s.type(torch.FloatTensor))
      loss = model.forward(i_q, i_a, i_s)
      loss.backward()
      average_gradients(model)
      optimizer.step()
      print "Training loss", loss.data.sum()
      # cnt += 1
      # if cnt % config.valid_every == 0:
      #     loss = model.forward(dev_q_t, dev_a_t, dev_sim_t)
      #     print "Validation loss", loss.data.sum()

   pre_embed_file = config.pre_embed_file+'_train_no_batch_'+str(rank)
   model.save(pre_embed_file)

   '''
   num_batches = math.ceil(len(train_set.dataset) / float(bsz))
   for epoch in range(10):
       epoch_loss = 0.0
       for data, target in train_set:
           #data, target = torch.autograd.Variable(data), torch.autograd.Variable(target)
           data, target = torch.autograd.Variable(data.cuda(gpu_rank)), torch.autograd.Variable(target.cuda(gpu_rank))
           optimizer.zero_grad()
           output = model(data)
           loss = torch.nn.functional.nll_loss(output, target)
           epoch_loss += loss.item()
           loss.backward()
           average_gradients(model)
           optimizer.step()

       print('Rank ', torch.distributed.get_rank(), ' on ', hostname, ' at epoch ', epoch, ': ', epoch_loss / num_batches)
   '''

if __name__ == "__main__":

   slurmjobid = os.environ['SLURM_JOB_ID']
   ntasks = int(os.environ['SLURM_NTASKS'])
   rank_id = int(os.environ['SLURM_PROCID'])

   print('job_id', slurmjobid)
   print('ntasks', ntasks)
   print('rank_id', rank_id)
   torch.distributed.init_process_group(backend='gloo', init_method='file:///home/neesittg/tmp_scratch/kvmm/pytorch-shared-filesystem-init.o' + slurmjobid, rank=rank_id, world_size=ntasks)

   run(rank_id, ntasks)
