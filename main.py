import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.optim import SGD 

from torch_scatter import scatter 
from torchkge.utils.datasets import load_fb15k237

import numpy as np 

import os 

from layers import * 
from loss import * 
from evaluation import * 
from utils import * 
from dataloader import * 

# %%

data_path = "/home/sai/code/relation-prediction-3/data/FB15k-237"
with open(os.path.join(data_path, 'entities.dict')) as fin:
    entity2id = dict()
    for line in fin:
        eid, entity = line.strip().split('\t')
        entity2id[entity] = int(eid)

with open(os.path.join(data_path, 'relations.dict')) as fin:
    relation2id = dict()
    for line in fin:
        rid, relation = line.strip().split('\t')
        relation2id[relation] = int(rid)