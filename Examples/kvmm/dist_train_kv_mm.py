
import socket
import operator
from config import config
from utils import *
import pdb
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class KVMemoryReader(nn.Module):
    def __init__(self, config):
        super(KVMemoryReader, self).__init__()
        self.config = config
        self.embed_A = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_B = nn.Embedding(config.n_embed, config.d_embed)
        self.embed_C = nn.Embedding(config.n_embed, config.d_embed)
        self.H = nn.Linear(config.d_embed, config.d_embed)

        # self.register_buffer("embed_A", embed_A)
        # self.register_buffer("embed_B", embed_B)
        # self.register_buffer("embed_C", embed_C)

    def forward(self, qu, key, value, cand):
        # qu = Variable(qu)
        # key = Variable(key)
        # value = Variable(value)
        # cand = Variable(cand)
        embed_q = self.embed_B(qu)
        embed_w1 = self.embed_A(key)
        embed_w2 = self.embed_C(value)
        embed_c = self.embed_C(cand)

        #pdb.set_trace()
        q_state = torch.sum(embed_q, 1).squeeze(1)
        w1_state = torch.sum(embed_w1, 1).squeeze(1)
        w2_state = embed_w2

        for _ in range(self.config.hop):
            sent_dot = torch.mm(q_state, torch.transpose(w1_state, 0, 1))
            sent_att = F.softmax(sent_dot)

            a_dot = torch.mm(sent_att, w2_state)
            a_dot = self.H(a_dot)
            q_state = torch.add(a_dot, q_state)

        f_feat = torch.mm(q_state, torch.transpose(embed_c, 0, 1))
        score = F.log_softmax(f_feat)
        return score

    def predict(self, q, key, value, cand):
        score = self.forward(q, key, value, cand)
        _, index = torch.max(score.squeeze(0), 0)
        return index.data.numpy()[0]

    def load_embed(self, filename):
        embed_t = None
        with open(filename) as f:
            embed_t = torch.load(f)
        self.embed_A.weight = embed_t
        self.embed_B.weight = embed_t
        self.embed_C.weight = embed_t


def modify(q, wiki, pos, ans):
    tL = torch.LongTensor
    ret_q = []
    ret_key = []
    ret_value = []
    ret_cand = []
    ret_a = []
    for qu, w, p, a_ in zip(q, wiki, pos, ans):
        # encoding the candidate
        can_dict = {}
        qu = qu.numpy()
        w = w.numpy()
        p = p.numpy()
        a_ = a_.numpy()

        # generate local candidate
        len_w = len(w)
        cand_ind = []
        for i in range(len_w):
            if w[i][1] not in can_dict:
                can_dict[w[i][1]] = len(can_dict)
            if w[i][p[i]] not in can_dict:
                can_dict[w[i][p[i]]] = len(can_dict)
        if a_[0] not in can_dict:
            continue
        else:
            sort_l = sorted(can_dict.items(), key=operator.itemgetter(1))
            cand_l = [x[0] for x in sort_l]

            # split into key value format
            # pdb.set_trace()
            key_m, val_m = transKV(w, p)
            ret_q.append(tL(qu))
            ret_key.append(tL(key_m))
            ret_value.append(tL(val_m))
            ret_cand.append(tL(cand_l))
            ret_a.append(tL([can_dict[a_[0]]]))
    print len(ret_q) / len(q)
    return ret_q, ret_key, ret_value, ret_cand, ret_a


def transKV(sents, pos):
    unk = 2
    ret_k = []
    ret_v = []
    for sent, p in zip(sents, pos):
        k_ = sent[3:].tolist() + [unk]
        v_ = sent[1]
        # pdb.set_trace()
        ret_k.append(k_)
        ret_v.append(v_)
        # print toSent(k_),toSent([v_])

        k_ = [sent[1]] + sent[3:].tolist()
        v_ = sent[p]
        ret_k.append(k_)
        ret_v.append(v_)
        # print toSent(k_),toSent([v_])
    return np.array(ret_k), np.array(ret_v)


def adjust_learning_rate(optimizer, epoch):
    lr = config.lr / (2 ** (epoch // 10))
    print "Adjust lr to ", lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def eval(model, dev_q, dev_key, dev_value, dev_cand, dev_a):
    cnt = 0
    for i_q, i_k, i_v, i_cand, i_a in zip(dev_q, dev_key, dev_value, dev_cand, dev_a):
        i_q = i_q.unsqueeze(0)  # add dimension
        try:
            ind = model.predict(i_q, i_k, i_v, i_cand)
        except:
            continue
        if ind == i_a[0]:
            cnt += 1
    return cnt / len(dev_q)


def partition_dataset():
    train_q, train_w, train_e_p, train_a = load_from_file("./pkl/reader/{}/train_pair.pkl".format(config.d_embed))
    train_q, train_key, train_value, train_cand, train_a = modify(train_q, train_w, train_e_p, train_a)
    print 'data loaded...'

    rank_num = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    data_size = len(train_q)
    dist_size = int(data_size/world_size) + 1
    print 'data size', data_size
    print 'dist size', dist_size
    start_index = rank_num * dist_size
    end_index = min((rank_num + 1) * dist_size, data_size)

    print "Rank {0}, {1} batch expected".format(rank_num, len(train_q[start_index:end_index]) * config.epoch / config.batch_size)

    return train_q[start_index:end_index], train_key[start_index:end_index], train_value[start_index:end_index], \
           train_cand[start_index:end_index], train_a[start_index:end_index]

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

   gpu_rank = rank % 4
   print('gpu_rank = ', gpu_rank, ' rank = ', rank)

   dev_q, dev_key, dev_value, dev_cand, dev_a = None, None, None, None, None
   # if rank == 0:
   dev_q, dev_w, dev_e_p, dev_a = load_from_file("./pkl/reader/{}/dev_pair.pkl".format(config.d_embed))
   dev_q, dev_key, dev_value, dev_cand, dev_a = modify(dev_q, dev_w, dev_e_p, dev_a)

   batch_size = config.batch_size
   model = KVMemoryReader(config).cuda(gpu_rank)
   model.load_embed('./model/300/embedding.pre_train_'+str(rank))
   # here lr is divide by batch size since loss is accumulated
   optimizer = optim.SGD(model.parameters(), lr=config.lr)
   print "Training setting: lr {0}, batch size {1}".format(config.lr, batch_size)

   loss_function = nn.NLLLoss()

   # print "{} batch expected".format(len(dataset[0]) * config.epoch / batch_size)
   # train_q, train_key, train_value, train_cand, train_a = modify(dataset[0], dataset[1], dataset[2], dataset[3])
   train_q = dataset[0]
   train_key = dataset[1]
   train_value = dataset[2]
   train_cand = dataset[3]
   train_a = dataset[4]

   for e_ in range(config.epoch):
       if (e_ + 1) % 10 == 0:
           adjust_learning_rate(optimizer, e_)
       cnt = 0
       loss = Variable(torch.Tensor([0]).cuda(gpu_rank))
       for i_q, i_k, i_v, i_cand, i_a in zip(train_q, train_key, train_value, train_cand, train_a):
           cnt += 1
           i_q = i_q.unsqueeze(0)  # add dimension
           i_q, i_k, i_v, i_cand = Variable(i_q.cuda(gpu_rank)), Variable(i_k.cuda(gpu_rank)), \
                                   Variable(i_v.cuda(gpu_rank)), Variable(i_cand.cuda(gpu_rank))
           model.zero_grad()
           probs = model.forward(i_q, i_k, i_v, i_cand)
           i_a = Variable(i_a.cuda(gpu_rank))
           curr_loss = loss_function(probs, i_a)
           loss = torch.add(loss, torch.div(curr_loss, batch_size))

           # naive batch implemetation, the lr is divided by batch size
           if cnt % batch_size == 0:
               print 'Rank ', torch.distributed.get_rank(), ' on ', hostname, ' at epoch ', e_, "Training loss", loss.data.sum()
               loss.backward()
               average_gradients(model)
               optimizer.step()
               loss = Variable(torch.Tensor([0]).cuda(gpu_rank))
               model.zero_grad()
           if cnt % config.valid_every == 0:
               # if rank == 0:
               print 'Rank ', torch.distributed.get_rank(), " Accuracy:", eval(model, dev_q, dev_key, dev_value, dev_cand, dev_a)

   reader_file = config.reader_model+'_'+str(rank)
   dump_to_file(model, reader_file)


if __name__ == "__main__":

   slurmjobid = os.environ['SLURM_JOB_ID']
   ntasks = int(os.environ['SLURM_NTASKS'])
   rank_id = int(os.environ['SLURM_PROCID'])

   print('job_id', slurmjobid)
   print('ntasks', ntasks)
   print('rank_id', rank_id)
   torch.distributed.init_process_group(backend='gloo', init_method='file:///home/neesittg/tmp_scratch/kvmm/pytorch-shared-filesystem-init.o' + slurmjobid, rank=rank_id, world_size=ntasks)

   run(rank_id, ntasks)
