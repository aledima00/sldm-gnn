import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import GATv2Conv as _GATv2Conv, global_mean_pool as _gmean_pool, global_max_pool as _gmax_pool
import torch.nn.functional as _F
from typing import Literal as _Lit

class GRUGAT(_nn.Module):
    def __init__(self, dynamic_features_num, has_dims, gru_hidden_size=128,gru_num_layers=1, gat_edge_fnum:int=None, gat_edge_aggregated:bool=True, gat_inner_dims=[96, 96], gat_nheads=4,fc_dims1=[128, 64], fc_dims2=[50,50], out_dim=10, num_st_types=256, emb_dim=12,*, negative_slope=0.2, dropout=0.6):
        super().__init__()
        

        assert len(gat_inner_dims) >= 1, "gat_inner_dims must contain at least one element"
        assert gat_nheads >= 1, "gat nheads must be at least 1"
        assert gat_edge_fnum is None or gat_edge_fnum >= 0, "gat_edge_fnum must be non-negative or None"
        self.gat_edge_fnum = gat_edge_fnum

        #TODO Implement additional edge-wise GRU to support non-aggregated edges case
        if not gat_edge_aggregated:
            raise NotImplementedError("Currently only gat_edge_aggregated=True is supported")

        # 1. GRU layer to process dynamic features
        # x shape: [batch_vnum, framenum, feature_dim]
        self.gru = _nn.GRU(
            input_size=dynamic_features_num,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )

        # 2. embedding for station types
        self.st_emb = _nn.Embedding(num_st_types, emb_dim)

        # 3. concat all input features
        last_step_dims = gru_hidden_size + (2 if has_dims else 0) + emb_dim

        # 4. fully connected layers before GAT
        ldims1 = [last_step_dims] + fc_dims1
        self.fc1s = _nn.ModuleList([
            _nn.Sequential(
                _nn.Linear(ldims1[i], ldims1[i+1]),
                _nn.ReLU(),
                _nn.Dropout(p=0.1)
            ) for i in range(len(ldims1)-1)
        ])
        last_step_dims = ldims1[-1]


        # 5. GAT layers
        gdims = [last_step_dims] + gat_inner_dims
        self.convs = _nn.ModuleList([
            _GATv2Conv(
                in_channels=gdims[i]*(1 if i==0 else gat_nheads),
                out_channels=gdims[i+1],
                heads=gat_nheads,
                concat=True,
                negative_slope=negative_slope,
                edge_dim=gat_edge_fnum,
                dropout=dropout
            ) for i in range(len(gdims)-1)
        ])

        self.norms = _nn.ModuleList([
            _nn.LayerNorm(gdims[i+1]*gat_nheads) for i in range(len(gdims)-1)
        ])

        last_step_dims = gdims[-1]

        # 6. final fc layers
        ldims = [last_step_dims*gat_nheads*2] + fc_dims2 + [out_dim]
        self.fc2s = _nn.ModuleList([
            *[ _nn.Sequential(
                _nn.Linear(ldims[i], ldims[i+1]),
                _nn.ReLU(),
                _nn.Dropout(p=dropout)
            ) for i in range(len(ldims)-2)],
            _nn.Linear(ldims[-2], ldims[-1])
        ])
        self.dropout = dropout

    def forward(self, data):
        x, edge_index, edge_attr, xdims, xsttype, batch = data.x, data.edge_index, data.edge_attr, data.xdims, data.xsttype, data.batch
        st_embedded:_torch.Tensor = self.st_emb(xsttype)

        # process dynamic features with GRU
        # here data are in shape [batch_vnum, seq_len, feature_dim]
        # get the last hidden state
        _, hlast = self.gru(x)
        hlast = hlast[-1, :, :] # take last layer's hidden state
        # shape [batch_vnum, gru_hidden_size]

        x = _torch.cat([hlast, xdims, st_embedded], dim=1)

        # pass through fc layers before GAT
        for fc in self.fc1s:
            x = fc(x)
        


        if self.gat_edge_fnum is not None:
            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, edge_index, edge_attr)
                x = norm(x)
                x = _F.elu(x)
                x = _F.dropout(x, p=self.dropout, training=self.training)
        else:
            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = _F.elu(x)
                x = _F.dropout(x, p=self.dropout, training=self.training)

        xmean = _gmean_pool(x, batch)
        xmax = _gmax_pool(x, batch)
        x = _torch.cat([xmean, xmax], dim=1)

        # now fc layers
        for fc in self.fc2s:
            x = fc(x)
        return x