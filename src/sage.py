import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool as _gmean_pool, global_max_pool as _gmax_pool
import torch.nn.functional as F
from typing import Literal as _Lit

class SageBlock(nn.Module):
    def __init__(self, hdims:list[int], dropout:float|None=None, negative_slope:float|None=None):
        super().__init__()
        assert len(hdims) >= 1, "hdims must contain at least one element"
        self.convs = nn.ModuleList([SAGEConv(hdims[i], hdims[i+1]) for i in range(len(hdims)-1)])
        self.posts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hdims[i+1]),
                nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else nn.ReLU(),
                nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
            ) for i in range(len(hdims)-1)
        ])
    def forward(self, x, edge_index):
        for conv, post in zip(self.convs, self.posts):
            x = conv(x, edge_index)
            x = post(x)
        return x

class GruSage(nn.Module):
    def __init__(self, dynamic_features_num:int, has_dims:bool, frames_num:int, gru_hidden_size:int, gru_num_layers:int, fc1dims:list[int], sage_hidden_dims:list[int]=[128, 128], fc2dims:list[int]=[50,50], out_dim:int=1, num_st_types:int=256, emb_dim:int=12, dropout:float|None=None, negative_slope:float|None=None, global_pooling:_Lit['mean', 'max','double']='double'):
        super().__init__()

        #TODO validate inputs
        assert len(sage_hidden_dims) >= 1, "sage_hidden_dims must contain at least one element"
        # ...

        # 1 - embedding for station types
        self.st_emb = nn.Embedding(num_st_types, emb_dim)

        # 2 - GRU layer to process dynamic features
        self.gru = nn.GRU(
            input_size=dynamic_features_num,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )

        # 3. concat all input features
        last_step_dims = gru_hidden_size + (2 if has_dims else 0) + emb_dim

        # 4 - fully connected layers before GraphSAGE
        ldims1 = [last_step_dims] + fc1dims
        self.fc1s = nn.ModuleList([
            nn.Sequential(
                nn.Linear(ldims1[i], ldims1[i+1]),
                nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else nn.ReLU(),
                nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
            ) for i in range(len(ldims1)-1)
        ])
        last_step_dims = ldims1[-1]

        # 5 - GraphSAGE layers
        sagedims = [last_step_dims] + sage_hidden_dims
        self.sage = SageBlock(sagedims, dropout=dropout, negative_slope=negative_slope)
        last_step_dims = sagedims[-1]

        # 6 - global pooling
        match global_pooling:
            case 'mean':
                self.global_pool = _gmean_pool
            case 'max':
                self.global_pool = _gmax_pool
            case 'double':
                self.global_pool = lambda x,batch: torch.cat([_gmean_pool(x, batch), _gmax_pool(x, batch)], dim=1)
                last_step_dims = last_step_dims * 2
            case _:
                raise ValueError(f"Unsupported global_pooling method: {global_pooling}")


        # 7 - fully connected layers after GraphSAGE
        ldims = [last_step_dims] + fc2dims
        self.fc2s = nn.ModuleList([
            *[nn.Sequential(
                nn.Linear(ldims[i], ldims[i+1]),
                nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else nn.ReLU(),
                nn.Dropout(p=dropout) if dropout is not None else nn.Identity()
            ) for i in range(len(ldims)-1)]
        ])

        # 8 - final output layer
        self.linout = nn.Linear(ldims[-1], out_dim)

        self.dropout = dropout
        self.negative_slope = negative_slope

    def forward(self, data):
        x, edge_index, edge_attr, xdims, xsttype, batch = data.x, data.edge_index, data.edge_attr, data.xdims, data.xsttype, data.batch

        # 1 - embedding for station types
        st_embedded:torch.Tensor = self.st_emb(xsttype)

        # 2 - process dynamic features with GRU
        # x shape: [batch_vnum, framenum, feature_dim]
        gru_out, hlast = self.gru(x)
        x = hlast[-1] # take last hidden state

        # 3 - concat all input features
        x = torch.cat([x, xdims,st_embedded], dim=1)

        # 4 - fc layers before GraphSAGE
        for fc in self.fc1s:
            x = fc(x)

        # 5 - GraphSAGE layers
        x = self.sage(x, edge_index)

        # 6 - graph level readout
        x = self.global_pool(x, batch)

        # 7 - fc layers after GraphSAGE
        for fc in self.fc2s:
            x = fc(x)
        # 8 - final output layer
        x = self.linout(x)
        return x
    
    def grads(self):
        per_layer_grads = dict()

        def addGradsToDict(module:nn.Module, name:str):
            buf = [param.grad.view(-1) for pname, param in module.named_parameters() if (param.requires_grad and param.grad is not None)]
            per_layer_grads[name] = torch.cat(buf) if len(buf) > 0 else None

        addGradsToDict(self.st_emb, 'StType Embedding')
        addGradsToDict(self.gru, 'GRU Layer')
        addGradsToDict(self.fc1s, 'FC Layers before SAGE')
        addGradsToDict(self.sage, 'GraphSAGE Layers')
        addGradsToDict(self.fc2s, 'FC Layers after SAGE')
        addGradsToDict(self.linout, 'Final Output Layer')

        # concat all grads
        tot_grads = torch.cat([g for g in per_layer_grads.values() if g is not None]) if len(per_layer_grads) > 0 else None

        tot_norm = tot_grads.norm().item() if tot_grads is not None else None
        layer_norms ={
            name: (gr.norm().item() if gr is not None else None)
            for name, gr in per_layer_grads.items()
        }
        return tot_norm, layer_norms