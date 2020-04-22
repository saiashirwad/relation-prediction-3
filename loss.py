import torch 
import torch.nn as nn 
import numpy as np 


def loss_transe(triplets, neg_sampling_ratio, ent_embed, rel_embed, device='cpu'):
    """
    Triplets order: src, dst, rel
    """
    n = len(triplets)
    if type(triplets) == np.ndarray:
        triplets = torch.from_numpy(triplets)

    pos_triplets = triplets[:n // (neg_sampling_ratio + 1)]
    pos_triplets = torch.cat([pos_triplets for _ in range(neg_sampling_ratio)])

    neg_triplets = triplets[n // (neg_sampling_ratio + 1):]


    src_embed_ = ent_embed[pos_triplets[:, 0]]
    dst_embed_ = ent_embed[pos_triplets[:, 1]]
    rel_embed_ = rel_embed[pos_triplets[:, 2]]

    x = src_embed_ + rel_embed_ - dst_embed_
    pos_norm = torch.norm(x, p=2, dim=1)


    src_embed_ = ent_embed[neg_triplets[:, 0]]
    dst_embed_ = ent_embed[neg_triplets[:, 1]]
    rel_embed_ = rel_embed[neg_triplets[:, 2]]

    x = src_embed_ + rel_embed_ - dst_embed_
    neg_norm = torch.norm(x, p=2, dim=1)

    y = torch.ones(len(pos_triplets)).to(device)

    loss_fn = nn.MarginRankingLoss(margin=5)
    loss = loss_fn(pos_norm, neg_norm, y)

    return loss