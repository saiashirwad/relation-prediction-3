import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader

from tqdm import tqdm
import os 

from utils import negative_sampling

class Trainer:
    def __init__(self, name, model: nn.Module, kg, n_epochs=1000, batch_size=2000, device="cuda", 
        optim = "sgd", lr = 0.001, checkpoint_dir="checkpoints"):
        self.name = name
        
        self.work_threads = 4 
        self.lr = lr 
        self.weight_decay = None

        self.n_epochs = n_epochs

        self.model = model
        self.optim = optim.SGD(self.model.parameters(), lr)
        self.dataloader = DataLoader(kg, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

        self.n_ent, self.n_rel = kg.n_ent, kg.n_rel 

        self.negative_rate = 10
    
    def train_one_step(self, data):
        self.model.zero_grad() 
        loss = self.model(data, "tail")
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def run(self):
        self.model.train()
        if self.optim == "sgd":
            self.optimizer = optim.SGD(self.model.parameters, lr=self.lr)

        training_range = tqdm(range(self.n_epochs))
        for epoch in training_range:
            res = 0
            for batch in self.dataloader:
                triplets = torch.stack(batch)
                triplets, _ = negative_sampling(triplets, self.n_ent, self.negative_rate)
                triplets = triplets.to(self.device)

                loss = self.train_one_step(triplets, "tail")
                res += loss 
            training_range.set_description("Epoch %d | loss: %f" % (epoch, res))

            if epoch % 10 == 0:
                if "checkpoints" not in os.listdir("."):
                    os.mkdir("checkpoint")
                torch.save(self.model.state_dict(), os.path.join(f"./checkpoints/{self.name}"))