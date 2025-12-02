import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
import torch.nn.functional as F

class GraphSAGEGraphLevel(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128, 128], out_dim=10, num_st_types=256, emb_dim=12, dropout=0.1):
        super().__init__()
        self.st_emb = nn.Embedding(num_st_types, emb_dim)
        dims = [in_dim + emb_dim] + hidden_dims
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i+1]) for i in range(len(hidden_dims))
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for dim in hidden_dims
        ])
        self.lin = nn.Linear(dims[-1]*2, out_dim)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, xdims, xsttype, batch = data.x, data.edge_index, data.edge_attr, data.xdims, data.xsttype, data.batch
        st_embedded:torch.Tensor = self.st_emb(xsttype)
        x = torch.cat([x, xdims,st_embedded], dim=1)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        xmean = global_mean_pool(x, batch)
        xmax = global_max_pool(x, batch)
        x = torch.cat([xmean, xmax], dim=1)
        logits = self.lin(x)
        return logits