import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from torch.optim import SGD

from layers import KGNet
from loss import loss_transe
from utils import negative_sampling

def train(kg_train, in_dim, out_dim, negative_rate, batch_size, device, n_epochs, lr=0.001):
    dataloader = DataLoader(kg_train, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
    batches = [b for b in dataloader]
    n_ent, n_rel = kg_train.n_ent, kg_train.n_rel 

    model = KGNet(n_ent, n_rel, in_dim, out_dim, 0.5)
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
        
        print(f"epoch: {epoch}, loss: {sum(losses) / len(losses)}")