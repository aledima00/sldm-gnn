import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import global_mean_pool as _gmean_pool, global_max_pool as _gmax_pool
from typing import Literal as _Lit

from .blocks.gatblock import GATv2Block as _Gatv2Block

class GRUGAT(_nn.Module):
    def __init__(self, dynamic_features_num:int, has_dims:bool, frames_num:int,emb_dim:int, gru_hidden_size:int,gru_num_layers:int,fc1dims:list[int], gat_edge_fnum:int|None, gat_edge_aggregated:bool, gat_inner_dims:list[int], gat_nheads:int, fc2dims:list[int], out_dim:int=1, num_st_types:int=256, *, dropout:float|None=None, negative_slope:float|None=None, gat_concat:bool=True, global_pooling:_Lit['mean', 'max','double']='double'):
        super().__init__()
        

        assert len(gat_inner_dims) >= 1, "gat_inner_dims must contain at least one element"
        assert gat_nheads >= 1, "gat nheads must be at least 1"
        assert gat_edge_fnum is None or gat_edge_fnum >= 0, "gat_edge_fnum must be non-negative or None"
        # TODO finish parameter validation

        self.gat_edge_fnum = gat_edge_fnum

        #TODO Implement additional edge-wise GRU to support non-aggregated edges case
        if not gat_edge_aggregated:
            raise NotImplementedError("Currently only gat_edge_aggregated=True is supported")

        # 1. embedding for station types
        self.st_emb = _nn.Embedding(num_st_types, emb_dim)
        
        # 2. GRU layer to process dynamic features
        # x shape: [batch_vnum, framenum, feature_dim]
        self.gru = _nn.GRU(
            input_size=dynamic_features_num,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )

        
        # 3. concat all input features
        last_step_dims = gru_hidden_size + (2 if has_dims else 0) + emb_dim

        # 4. fully connected layers before GAT
        ldims1 = [last_step_dims] + fc1dims
        self.fc1s = _nn.ModuleList([
            _nn.Sequential(
                _nn.Linear(ldims1[i], ldims1[i+1]),
                _nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else _nn.ReLU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(ldims1)-1)
        ])
        last_step_dims = ldims1[-1]


        # 5. GAT layers
        gdims = [last_step_dims] + gat_inner_dims
        self.gat = _Gatv2Block( gdims, gat_nheads, gat_concat,
            gat_edge_fnum=gat_edge_fnum,
            dropout=dropout,
            negative_slope=negative_slope
        )
        last_step_dims = gdims[-1]

        # 6. global pooling
        match global_pooling:
            case 'mean':
                self.global_pool = _gmean_pool
            case 'max':
                self.global_pool = _gmax_pool
            case 'double':
                self.global_pool = lambda x,batch: _torch.cat([_gmean_pool(x, batch), _gmax_pool(x, batch)], dim=1)
                last_step_dims = last_step_dims * 2
            case _:
                raise ValueError(f"Unsupported global_pooling method: {global_pooling}")


        # 7. final fc layers
        ldims = [last_step_dims] + fc2dims
        self.fc2s = _nn.ModuleList([
            *[ _nn.Sequential(
                _nn.Linear(ldims[i], ldims[i+1]),
                _nn.ReLU(),
                _nn.Dropout(p=dropout)
            ) for i in range(len(ldims)-1)],
        ])

        # 8. final output layer
        self.linout = _nn.Linear(ldims[-1], out_dim)

        self.dropout = dropout
        self.negative_slope = negative_slope
        self.init_weights()

    def init_weights(self):
        # initialize GRU params
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                _torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                _torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                _torch.nn.init.zeros_(param.data)

        # initialize embedding params
        _torch.nn.init.normal_(self.st_emb.weight, mean=0.0, std=0.01)

        # initialize GAT params
        for conv in self.convs:
            _torch.nn.init.xavier_uniform_(conv.lin_l.weight)
            _torch.nn.init.xavier_uniform_(conv.lin_r.weight)
            if conv.edge_dim is not None:
                _torch.nn.init.xavier_uniform_(conv.lin_edge.weight)
            if conv.bias is not None:
                _torch.nn.init.zeros_(conv.bias)

        for fc in self.fc1s:
            _torch.nn.init.xavier_uniform_(fc[0].weight)
            _torch.nn.init.zeros_(fc[0].bias)

        # initialize fc params
        for fc in self.fc2s:
            if isinstance(fc, _nn.Sequential):
                _torch.nn.init.xavier_uniform_(fc[0].weight)
                _torch.nn.init.zeros_(fc[0].bias)
            else:
                _torch.nn.init.xavier_uniform_(fc.weight)
                _torch.nn.init.zeros_(fc.bias)            

        
    def forward(self, data):
        x, edge_index, edge_attr, xdims, xsttype, batch = data.x, data.edge_index, data.edge_attr, data.xdims, data.xsttype, data.batch

        # 1 - embedding for station types
        st_embedded:_torch.Tensor = self.st_emb(xsttype)

        # 2 - process dynamic features with GRU
        # x shape: [batch_vnum, framenum, feature_dim]
        # get the last hidden state
        _, hlast = self.gru(x)
        x = hlast[-1, :, :] # take last layer's hidden state
        # shape [batch_vnum, gru_hidden_size]

        # 3 - concat all input features
        x = _torch.cat([x, xdims, st_embedded], dim=1)

        # 4 - fc layers before GAT
        for fc in self.fc1s:
            x = fc(x)


        # 5 - GAT layers
        x = self.gat(x, edge_index, edge_attr)

        # 6 - graph level readout
        x = self.global_pool(x, batch)

        # 7 - fc layers after GAT
        for fc in self.fc2s:
            x = fc(x)

        # 8 - final output layer
        x = self.linout(x)
        return x
    
    def grads(self):
        per_layer_grads = dict()

        def addGradsToDict(module:_nn.Module, name:str):
            buf = [param.grad.view(-1) for pname, param in module.named_parameters() if (param.requires_grad and param.grad is not None)]
            per_layer_grads[name] = _torch.cat(buf) if len(buf) > 0 else None

        addGradsToDict(self.st_emb, 'StType Embedding')
        addGradsToDict(self.gru, 'GRU Layer')
        addGradsToDict(self.fc1s, 'FC Layers before GAT')
        addGradsToDict(self.gat, 'GAT Layers')
        addGradsToDict(self.fc2s, 'FC Layers after GAT')
        addGradsToDict(self.linout, 'Final Output Layer')

        # concat all grads
        tot_grads = _torch.cat([g for g in per_layer_grads.values() if g is not None]) if len(per_layer_grads) > 0 else None
        
        tot_norm = tot_grads.norm().item() if tot_grads is not None else None
        layer_norms ={
            name: (gr.norm().item() if gr is not None else None)
            for name, gr in per_layer_grads.items()
        }
        return tot_norm, layer_norms