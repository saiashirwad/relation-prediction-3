import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader

from torchkge.utils.datasets import load_fb15k237

from tqdm import tqdm
import os 

from utils import negative_sampling

import evaluation

class Trainer:
    def __init__(self, name, model: nn.Module, n_epochs=1000, batch_size=2000, device="cuda", 
        optim_ = "sgd", lr = 0.001, checkpoint_dir="checkpoints"):
        self.name = name
        
        self.work_threads = 4 
        self.lr = lr 
        self.weight_decay = None

        self.n_epochs = n_epochs
        self.device = device

        self.kg_train, self.kg_test, self.kg_val = load_fb15k237()

        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(), lr)
        self.dataloader_train = DataLoader(self.kg_train, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
        self.dataloader_eval = DataLoader(self.kg_val, batch_size=1, shuffle=True, pin_memory=True)

        self.n_ent, self.n_rel = self.kg_train.n_ent, self.kg_train.n_rel 

        self.negative_rate = 10
    
    def train_one_step(self, data, mode="tail"):
        self.model.zero_grad() 
        loss = self.model(data, mode)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def run(self):
        self.model.train()
        training_range = tqdm(range(self.n_epochs))
        for epoch in training_range:
            res = 0
            for batch in self.dataloader_train:
                triplets = torch.stack(batch)
                triplets, _ = negative_sampling(triplets, self.n_ent, self.negative_rate)
                triplets = triplets.to(self.device)

                loss = self.train_one_step(triplets, "tail")
                res += loss 
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))

            # if epoch % 10 == 0:
            #     if "checkpoints" not in os.listdir("."):
            #         os.mkdir("checkpoint")
            #     torch.save(self.model.state_dict(), os.path.join(f"checkpoints/{self.name}"))
    
    def evaluate(self, n_samples=100):
        self.model.eval()
        evaluation.eval(self.kg_val, self.model, n_samples)
