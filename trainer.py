import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader

from tqdm import tqdm

from utils import negative_sampling

class Trainer:
    def __init__(self, model: nn.Module, kg, n_epochs=1000, batch_size=2000, device="cuda", 
        optim = "sgd", lr = 0.001, checkpoint_dir="checkpoints"):
        
        self.work_threads = 4 
        self.lr = lr 
        self.weight_decay = None

        self.n_epochs = n_epochs

        self.model = model
        self.optim = SGD(self.model.parameters(), lr)
        self.dataloader = DataLoader(kg, batch_size=batch_size, shuffle=False, pin_memory=torch.cuda.is_available())

        self.n_ent, self.n_rel = kg.n_ent, kg.n_rel 

        self.negative_rate = 10
    
    def train_one_step(self, data):
        self.model.zero_grad() 
        loss = self.model()
    
    def run(self):
        if self.optim == "sgd":
            self.optimizer = optim.SGD(self.model.parameters, lr=self.lr)
        

        training_range = tqdm(range(self.n_epochs))
        for epoch in training_range:
            for batch in self.dataloader:
                triplets = torch.stack(batch)
                triplets, labels = negative_sampling(triplets, self.n_ent, self.negative_rate)
                triplets, labels = triplets.to(self.device), labels.to(self.device)

                