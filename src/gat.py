import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import GATConv as _GATConv, global_mean_pool as _gmpool
import torch.nn.functional as _F
from typing import Literal as _Lit

class GATGraphLevel(_nn.Module):
    def __init__(self, in_dim, hidden_dims=[128, 128], out_dim=10, num_st_types=256, emb_dim=12,*, heads=4, negative_slope=0.2, dropout=0.6):
        super().__init__()
        self.st_emb = _nn.Embedding(num_st_types, emb_dim)
        dims = [in_dim + emb_dim] + hidden_dims
        self.convs = _nn.ModuleList([
            _GATConv(dims[i]*(1 if i==0 else heads), dims[i+1], heads=heads, concat=(i < len(hidden_dims)-1), negative_slope=negative_slope, dropout=dropout) for i in range(len(hidden_dims))
        ])
        self.lin = _nn.Linear(dims[-1], out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, st_types_feats, batch = data.x, data.edge_index, data.edge_attr, data.st_types_feats, data.batch
        st_embedded:_torch.Tensor = self.st_emb(st_types_feats)
        st_embedded = st_embedded.squeeze(1)
        x = _torch.cat([x, st_embedded], dim=1)
        for i,conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = _F.elu(x)
        x = _gmpool(x, batch)
        logits = self.lin(x)
        return logits
