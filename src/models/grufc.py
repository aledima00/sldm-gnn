import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import GATv2Conv as _GATv2Conv, global_mean_pool as _gmean_pool, global_max_pool as _gmax_pool
import torch.nn.functional as _F
from typing import Literal as _Lit


class GruFC(_nn.Module):
    def __init__(self, dynamic_features_num:int, has_dims:bool, frames_num:int, gru_hidden_size:int, gru_num_layers:int, fc_dims:list[int], out_dim:int=1, num_st_types=256, emb_dim=12,*,dropout=None,negative_slope=None, global_pooling:_Lit['mean', 'max','double']='double'):
        super().__init__()

        #TODO validate inputs
        # ...

        # 1. concat embedding for station types
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
        self.hasdims = has_dims

        # 4 - global pooling
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

        # 5. fully connected layers
        ldims = [last_step_dims] + fc_dims
        self.fcs = _nn.ModuleList([
            _nn.Sequential(
                _nn.Linear(ldims[i], ldims[i+1]),
                _nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else _nn.ReLU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(ldims)-1)
        ])

        # 6. output layer
        self.linout = _nn.Linear(ldims[-1], out_dim)

    #     self.init_weights()

    # def init_weights(self):
    #     # init GRU with xavier uniform for ih, orthogonal for hh and zeros for biases
    #     for name, param in self.gru.named_parameters():
    #         if 'weight_ih' in name:
    #             _nn.init.xavier_uniform_(param.data)
    #         elif 'weight_hh' in name:
    #             _nn.init.orthogonal_(param.data)
    #         elif 'bias' in name:
    #             _nn.init.zeros_(param.data)

    #     # init linear layers with kaiming uniform
    #     for fc in self.fcs:
    #         # linear - relu - dropout
    #         _nn.init.kaiming_uniform_(fc[0].weight, nonlinearity='relu')
    #         _nn.init.zeros_(fc[0].bias)
    #     _nn.init.kaiming_uniform_(self.linout.weight, nonlinearity='linear')
    #     _nn.init.zeros_(self.linout.bias)

    def forward(self, data):
        x, xsttype, batch = data.x, data.xsttype, data.batch

        # 1. embed station types
        st_embedded:_torch.Tensor = self.st_emb(xsttype)

        # 2. process dynamic features with GRU
        _,hlast = self.gru(x)
        x = hlast[-1,:,:]

        # 3. concat all input features
        if self.hasdims:
            xdims = data.xdims
            x = _torch.cat([x, xdims, st_embedded], dim=1)
        else:
            x = _torch.cat([x, st_embedded], dim=1)

        # 4. global pooling
        x = self.global_pool(x, batch)

        # 5. FC layers
        for fc in self.fcs:
            x = fc(x)

        # 6. output layer
        x = self.linout(x)
        return x
    
    def grads(self):
        per_layer_grads = dict()

        def addGradsToDict(module:_nn.Module, name:str):
            buf = [param.grad.view(-1) for pname, param in module.named_parameters() if (param.requires_grad and param.grad is not None)]
            per_layer_grads[name] = _torch.cat(buf) if len(buf) > 0 else None

        addGradsToDict(self.st_emb, 'StType Embedding')
        addGradsToDict(self.gru, 'GRU Layer')
        addGradsToDict(self.fcs, 'FC Layers')
        addGradsToDict(self.linout, 'Output Layer')

        tot_grads = _torch.cat([g for g in per_layer_grads.values() if g is not None]) if len(per_layer_grads) > 0 else None
        
        tot_norm = tot_grads.norm().item() if tot_grads is not None else None
        layer_norms ={
            name: (gr.norm().item() if gr is not None else None)
            for name, gr in per_layer_grads.items()
        }

        return tot_norm, layer_norms