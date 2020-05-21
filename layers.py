import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter

class SNAFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]

        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(torch.cuda.is_available()):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
        return None, grad_values, None, None, None

class SparseNeighborhoodAggregation(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SNAFunction.apply(edge, edge_w, N, E, out_features)


class KGLayer(nn.Module):
    def __init__(self, n_entities, n_relations, in_dim, out_dim, input_drop=0.5, concat=True, device="cuda"):
        super(KGLayer, self).__init__()

        self.n_entities = n_entities
        self.n_relations = n_relations
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device

        self.a = nn.Linear(3 * in_dim, out_dim).to(device)
        nn.init.xavier_normal_(self.a.weight.data, gain=1.414)

        self.concat = concat

        self.a_2 = nn.Linear(out_dim, 1).to(device)
        nn.init.xavier_normal_(self.a_2.weight.data, gain=1.414)

        self.sparse_neighborhood_aggregation = SparseNeighborhoodAggregation()

        self.ent_embed = nn.Embedding(n_entities, in_dim, max_norm=1, norm_type=2).to(device)
        self.rel_embed = nn.Embedding(n_relations, in_dim, max_norm=1, norm_type=2).to(device)

        nn.init.xavier_normal_(self.ent_embed.weight.data, 1.414)
        nn.init.xavier_normal_(self.rel_embed.weight.data, 1.414)

        self.input_drop = nn.Dropout(input_drop)

        self.bn0 = nn.BatchNorm1d(3 * in_dim).to(device)
        self.bn1 = nn.BatchNorm1d(out_dim).to(device)


    def forward(self, triplets, ent_embed=None, rel_embed=None):

        N = self.n_entities

        if ent_embed is None:
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
        else:
            h = torch.cat((
                ent_embed[triplets[:, 0]],
                ent_embed[triplets[:, 1]],
                rel_embed[triplets[:, 2]]
            ), dim=1)
            h_ = torch.cat((
                ent_embed[triplets[:, 1]],
                ent_embed[triplets[:, 0]],
               -rel_embed[triplets[:, 2]]
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
        h_rel_ = scatter(temp1[temp1.shape[0]//2 : , :], index=index, dim=0, reduce="mean")

        h_rel = h_rel - h_rel_  # add or subtract?

        if self.concat:
            return F.elu(h_ent), F.elu(h_rel)
        else:
            return h_ent, h_rel


class KGNet(nn.Module):
    def __init__(self, n_entities, n_relations, in_dim, out_dim, input_drop=0.5, n_layers=2, n_heads=5, device="cuda"):
        super(KGNet, self).__init__()

        self.n_heads = n_heads

        self.a1 = nn.ModuleList([
            KGLayer(n_entities, n_relations, in_dim, out_dim, input_drop, True, "cuda")
            for _ in range(n_heads)])
        self.a2 = KGLayer(n_entities, n_relations, n_heads * out_dim, out_dim, input_drop, False, "cuda")

    def forward(self, triplets):
        out = [a(triplets) for a in self.a1]
        h_ent = torch.cat([o[0] for o in out], dim=1)
        h_rel = torch.cat([o[1] for o in out], dim=1)

        return self.a2(triplets, h_ent, h_rel)
