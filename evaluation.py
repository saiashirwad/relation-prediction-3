import torch
from torch.utils.data import DataLoader

from utils import negative_sampling

def generate_eval_triplets(triplet, pos="head", n_ent=14541):
    if pos == "head":
        triplet = torch.tensor([triplet[1].item(), triplet[2].item()]).to(torch.long)
        triplets = triplet.repeat(n_ent).view(-1, 2).t()
        triplets = torch.stack((
            torch.arange(n_ent),
            triplets[0],
            triplets[1]
        ))
    elif pos == "tail":
        triplet = torch.tensor([triplet[0].item(), triplet[2].item()]).to(torch.long)
        triplets = triplet.repeat(n_ent).view(-1, 2).t()
        triplets = torch.stack((
            triplets[0],
            torch.arange(n_ent),
            triplets[1]
        ))

    return triplets.t()

def eval(kg_val, model, n_dim, n_samples):
    dataloader = DataLoader(kg_val, 1, shuffle=True)
    data = [d for d in dataloader]

    n = n_samples

    n_ent = kg_val.n_ent
    model.eval()

    head_rank_mean, tail_rank_mean = [0] * 2
    head_hits_10, tail_hits_10 = [0] * 2

    with torch.no_grad():
        for i in range(n_samples):
            triplets_h = generate_eval_triplets(data[i], "head", n_ent)
            triplets_h, _ = negative_sampling(triplets_h, n_ent, 0)
            triplets_h = triplets_h.to("cuda")
            ee, re = model(triplets_h)

            dst = ee[data[i][1]].squeeze()
            rel = re[data[i][2]].squeeze()
            dist = ee + (rel - dst).repeat(n_ent).view(-1, 100)
            head_preds = torch.topk(torch.norm(dist, dim=1), k=n_ent).indices.cpu().tolist()
            rank = head_preds.index(data[i][0])
            head_rank_mean += rank
            if rank < 10:
                head_hits_10 += 1

            # # # # tail
            triplets_t = generate_eval_triplets(data[i], "tail", n_ent)
            triplets_t, _, _, _ = negative_sampling(triplets_t, n_ent, 0)
            triplets_t = triplets_t.to("cuda")
            ee, re = model(triplets_t)

            src = ee[data[i][0]].squeeze()
            rel = re[data[i][2]].squeeze()
            dist = (src + rel).repeat(n_ent).view(-1, 100) - ee
            tail_preds = torch.topk(torch.norm(dist, dim=1), k=n_ent).indices.cpu().tolist()
            rank = tail_preds.index(data[i][1])
            tail_rank_mean += rank
            if rank < 10:
                tail_hits_10 += 1

        head_rank_mean /= n
        tail_rank_mean /= n
        head_hits_10 /= n
        tail_hits_10 /= n
        mean_rank = (head_rank_mean + tail_rank_mean) / 2
        hits_10 = (head_hits_10 + tail_hits_10) / 2

    #  print(f"Mean Rank: {mean_rank}")
    #  print(f"Hits@10: {hits0}")
    print("mean rank: {}".format(mean_rank))
    print("hits@10: {}".format(hits_10))
