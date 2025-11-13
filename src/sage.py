import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
import torch.nn.functional as F

class GraphSAGEGraphLevel(nn.Module):
    def __init__(self, in_dim, hidden_dims=[128, 128], out_dim=10, num_st_types=256, emb_dim=12):
        super().__init__()
        self.st_emb = nn.Embedding(num_st_types, emb_dim)
        # self.conv1 = SAGEConv(in_dim + emb_dim, hidden_dim)
        # self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        dims = [in_dim + emb_dim] + hidden_dims
        self.convs = nn.ModuleList([
            SAGEConv(dims[i], dims[i+1]) for i in range(len(hidden_dims))
        ])
        self.lin = nn.Linear(dims[-1], out_dim)

    def forward(self, data):
        x, edge_index, edge_attr, st_types_feats, batch = data.x, data.edge_index, data.edge_attr, data.st_types_feats, data.batch
        st_embedded:torch.Tensor = self.st_emb(st_types_feats)
        st_embedded = st_embedded.squeeze(1)
        x = torch.cat([x, st_embedded], dim=1)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        logits = self.lin(x)
        return logits