#! /usr/bin/python3.7
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from torchkge.utils.datasets import load_fb15k237

from layers import KGNet
from loss import loss_transe
from utils import negative_sampling
from rotate import RotAtt

def train_kgatt(kg_train, in_dim, out_dim, negative_rate, batch_size, device, n_epochs, lr=0.001, n_heads=10, model=None):
    dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    batches = [b for b in dataloader]
    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel
    if model is None:
        model = KGNet(n_ent, n_rel, in_dim, out_dim, 0.5, n_heads=n_heads)

    optimizer = SGD(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
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

        print("epoch: {}, loss: {}".format(epoch, (sum(losses) / len(losses))))

    return model

def train_rotatt():
    kg_train, kg_test, kg_val = load_fb15k237()
    in_dim = 100
    out_dim = 100 
    negative_rate = 10
    batch_size = 2000 
    device = "cuda"
    lr = 0.001 
    n_heads = 2 
    dropout = 0.5

    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel 
    model = RotAtt(n_ent, n_rel, in_dim, out_dim, n_heads, dropout, negative_rate, 6.0, 2.0, device)
    pass 