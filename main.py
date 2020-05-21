import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.optim import SGD 

from torch_scatter import scatter 
from torchkge.utils.datasets import load_fb15k237

import numpy as np 

from layers import * 
from loss import * 
from evaluation import * 
from utils import * 

import time 

import IPython

batch_size = 5000
in_dim = 50
out_dim = 50
negative_rate = 10 
device = "cuda"
n_epochs = 100 
lr = 0.001 
n_heads = 5 

kg_train, kg_test, kg_val = load_fb15k237() 
dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
batches = [b for b in dataloader]
n_ent, n_rel = kg_train.n_ent, kg_train.n_rel
model = KGNet(n_ent, n_rel, in_dim, out_dim, 0.5, 1, 1, "cuda")
optimizer = SGD(model.parameters(), lr=lr)
for epoch in range(1000):
    start = time.time()
    losses = []
    for i in range(len(batches)):
        batch = batches[i]
        triplets = torch.stack(batch)
        triplets, labels = negative_sampling(triplets, n_ent, negative_rate)
        triplets, labels = triplets.to(device), labels.to(device)

        model.zero_grad()

        model.train()
        ent_embed, rel_embed = model(triplets)
        loss = loss_transe(triplets, negative_rate, ent_embed, rel_embed, "cuda")
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print("epoch: {}, loss: {} ----- {}".format(epoch, (sum(losses) / len(losses)), time.time() - start))

IPython.embed()
