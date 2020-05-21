import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from torch_scatter import scatter

from layers import SparseNeighborhoodAggregation 


class RotAttLayer(nn.Module):
    def __init__(self, n_entities, n_relations, in_dim, out_dim, input_drop=0.5, 
                 margin=6.0, epsilon=2.0, device="cuda"):
        super().__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.loss = loss 

        self.a = nn.Linear(3 * in_dim, out_dim).to(device)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.concat = concat

        self.a_2 = nn.Linear(out_dim, 1).to(device)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

        self.sparse_neighborhood_aggregation = SparseNeighborhoodAggregation()
        
        self.ent_embed_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.out_dim]), 
            requires_grad = False
        )
        
        self.rel_embed_range = nn.Parameter(
            torch.Tensor([(self.margin + self.epsilon) / self.out_dim]),
            requires_grad = False
        )

        self.ent_embed = nn.Embedding(n_entities, in_dim, max_norm=1, norm_type=2).to(device)
        self.rel_embed = nn.Embedding(n_relations, in_dim, max_norm=1, norm_type=2).to(device)
        
        nn.init.uniform_(self.ent_embed.weight.data, -self.ent_embed_range.item(), self.ent_embed_range.item())
        nn.init.uniform_(self.rel_embed.weight.data, -self.rel_embed_range.item(), self.rel_embed_range.item())

        self.input_drop = nn.Dropout(input_drop)

        self.bn0 = nn.BatchNorm1d(3 * in_dim).to(device)
        self.bn1 = nn.BatchNorm1d(out_dim).to(device)
        
    def forward(self, triplets, eval=False, mode="head"):

        N = self.n_entities
        n = len(triplets)

        h = torch.cat((
            self.ent_embed(triplets[:, 0]),
            self.ent_embed(triplets[:, 1]),
            self.rel_embed(triplets[:, 2])
        ), dim=1)
        h_ = torch.cat((
            self.ent_embed(triplets[:, 1]),
            self.ent_embed(triplets[:, 0]),
            -self.rel_embed(triplets[:, 2])
        ), dim=1)

        h = torch.cat((h, h_))

        h = self.input_drop(self.bn0(h))
        c = self.bn1(self.a(h))
        b = -F.leaky_relu(self.a_2(c))
        e_b = torch.exp(b)

        temp = triplets.t()
        edges = torch.stack((
            torch.cat([temp[0], temp[1]]),
            torch.cat([temp[1], temp[0]])
        ))

        ebs = self.sparse_neighborhood_aggregation(edges, e_b, N, e_b.shape[0], 1)
        temp1 = e_b * c

        hs = self.sparse_neighborhood_aggregation(edges, temp1,  N, e_b.shape[0], self.out_dim)

        ebs[ebs == 0] = 1e-12
        h_ent = hs / ebs

        index = triplets[:, 2]
        h_rel  = scatter(temp1[ : temp1.shape[0]//2, :], index=index, dim=0, reduce="mean") 
        # h_rel_ =  scatter(temp1[temp1.shape[0]//2 : , :], index=index, dim=0, reduce="mean")

        return h_ent, h_rel
class RotAtt(nn.Module):
    def __init__(self, n_ent, n_rel, in_dim, out_dim, n_heads=1, input_drop=0.5, negative_rate = 10, margin=6.0, epsilon=2.0, device="cuda"):
        super().__init__() 

        self.n_heads = n_heads 
        self.device = device

        self.a = nn.ModuleList([
            RotAttLayer(
                n_ent, n_rel, in_dim, out_dim, input_drop, margin=margin, epsilon=epsilon
            )
        ] for _ in range(self.n_heads))

        self.ent_transform = nn.Linear(n_heads * out_dim, out_dim)
        self.rel_transform = nn.Linear(n_heads * out_dim, out_dim)

        self.pi = nn.Parameter(torch.Tensor([3.14159265358979323846])).to(device)
        self.pi.requires_grad = False 
        
        self.margin = margin
        self.epsilon = epsilon

        self.negative_rate = negative_rate

    def rotate(self, triplets, ent_embed, rel_embed, mode="head_batch"):
        
        h = ent_embed[triplets[:, 0]]
        t = ent_embed[triplets[:, 1]]
        r  = rel_embed[triplets[:, 2]]

        pi = self.pi

        re_head, im_head = torch.chunk(h, 2, dim=-1)
        re_tail, im_tail = torch.chunk(t, 2, dim=-1)

        phase_relation = r / (self.rel_embed_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        re_head = re_head.view(-1, re_relation.shape[0], re_head.shape[-1]).permute(1, 0, 2)
        re_tail = re_tail.view(-1, re_relation.shape[0], re_tail.shape[-1]).permute(1, 0, 2)
        im_head = im_head.view(-1, re_relation.shape[0], im_head.shape[-1]).permute(1, 0, 2)
        im_tail = im_tail.view(-1, re_relation.shape[0], im_tail.shape[-1]).permute(1, 0, 2)
        im_relation = im_relation.view(-1, re_relation.shape[0], im_relation.shape[-1]).permute(1, 0, 2)
        re_relation = re_relation.view(-1, re_relation.shape[0], re_relation.shape[-1]).permute(1, 0, 2)

        if mode == "head_batch":
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0).sum(dim = -1)
        return score.permute(1, 0).flatten()

    def forward(self, triplets, mode="tail_batch", eval=False):
        n = len(triplets)

        out = [a(triplets) for a in self.a]

        ent_embed = self.ent_transform(torch.cat([o[0] for o in out], dim=1))
        rel_embed = self.rel_transform(torch.cat([o[1] for o in out], dim=1))

        if not eval:
            pos_triplets = triplets[:n // (self.negative_rate + 1)]
            pos_triplets = torch.cat([pos_triplets for _ in range(self.negative_rate)])
            neg_triplets = triplets[n // (self.negative_rate + 1) :]

            pos_score = self.margin - self.rotate(pos_triplets, ent_embed, rel_embed, mode)
            neg_score = self.margin - self.rotate(neg_triplets, ent_embed, rel_embed, mode)

            y = torch.ones(len(pos_triplets))

            loss_fn = nn.MarginRankingLoss(margin=self.margin)
            loss = loss_fn(pos_score, neg_score, y)

            return loss 
        
        else:
            return self.margin - self.rotate(triplets, ent_embed, rel_embed, mode) 

    def predict(self, data):
        score = -self.forward(data, "tail_batch")
        return score.cpu().data.numpy()