import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
import torch.nn.functional as F

class GraphSAGEGraphLevel(nn.Module):
    def __init__(self, dynamic_features_num, has_dims, frames_num, hidden_dims=[128, 128], fcdims=[50,50], out_dim=10, num_st_types=256, emb_dim=12, dropout=0.1):
        super().__init__()

        # 1 - embedding for station types
        self.st_emb = nn.Embedding(num_st_types, emb_dim)

        # 2 - GraphSAGE layers
        in_dim = dynamic_features_num*frames_num + (2 if has_dims else 0) + emb_dim
        cdims = [in_dim] + hidden_dims
        self.convs = nn.ModuleList([
            SAGEConv(cdims[i], cdims[i+1]) for i in range(len(cdims)-1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(cdims[i+1]) for i in range(len(cdims)-1)
        ])

        # 3 - final fc layers
        ldims = [hidden_dims[-1]*2] + fcdims + [out_dim]
        self.fcs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ldims[i], ldims[i+1]),
                nn.ReLU(),
                nn.Dropout(p=dropout)
            ) for i in range(len(ldims)-1)
        ])
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
        # graph level readout
        xmean = global_mean_pool(x, batch)
        xmax = global_max_pool(x, batch)
        x = torch.cat([xmean, xmax], dim=1)
        # now fc layers
        for fc in self.fcs:
            x = fc(x)
        return x