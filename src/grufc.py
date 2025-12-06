import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import GATv2Conv as _GATv2Conv, global_mean_pool as _gmean_pool, global_max_pool as _gmax_pool
import torch.nn.functional as _F
from typing import Literal as _Lit


class GruFC(_nn.Module):
    def __init__(self, dynamic_features_num, has_dims, gru_hidden_size=128, gru_num_layers=1, fc_dims=[128, 64], out_dim=10, num_st_types=256, emb_dim=12,*,negative_slope=None, dropout=0.1):
        super().__init__()

        # 1. GRU layer to process dynamic features
        # x shape: [batch_vnum, framenum, feature_dim]
        self.gru = _nn.GRU(
            input_size=dynamic_features_num,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )

        # 2. concat embedding for station types
        self.st_emb = _nn.Embedding(num_st_types, emb_dim)

        # 3. concat all input features
        curdims = gru_hidden_size + (2 if has_dims else 0) + emb_dim
        self.hasdims = has_dims

        # 4. fully connected layers
        ldims = [curdims] + fc_dims
        self.fcs = _nn.ModuleList([
            _nn.Sequential(
                _nn.Linear(ldims[i], ldims[i+1]),
                _nn.ReLU() if negative_slope is None else _nn.LeakyReLU(negative_slope),
                _nn.Dropout(p=dropout)
            ) for i in range(len(ldims)-1)
        ])

        # 5. output layer
        self.linout = _nn.Linear(ldims[-1], out_dim)

        self.init_weights()

    def init_weights(self):
        # init GRU with xavier uniform for ih, orthogonal for hh and zeros for biases
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                _nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                _nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                _nn.init.zeros_(param.data)

        # init linear layers with kaiming uniform
        for fc in self.fcs:
            # linear - relu - dropout
            _nn.init.kaiming_uniform_(fc[0].weight, nonlinearity='relu')
            _nn.init.zeros_(fc[0].bias)
        _nn.init.kaiming_uniform_(self.linout.weight, nonlinearity='linear')
        _nn.init.zeros_(self.linout.bias)

    def forward(self, data):
        x, edge_index, edge_attr, xdims, xsttype, batch = data.x, data.edge_index, data.edge_attr, data.xdims, data.xsttype, data.batch
        # 0. embed station types
        st_embedded:_torch.Tensor = self.st_emb(xsttype)

        # 1.GRU
        _,hlast = self.gru(x)
        x = hlast[-1,:,:]

        # 2a. concat all features
        if self.hasdims:
            x = _torch.cat([x, xdims, st_embedded], dim=1)
        else:
            x = _torch.cat([x, st_embedded], dim=1)

        # 2.b global mean pooling
        x = _gmean_pool(x, batch)

        # 3. FC layers
        for fc in self.fcs:
            x = fc(x)

        # 4. output layer
        x = self.linout(x)
        return x