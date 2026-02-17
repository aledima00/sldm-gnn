import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import global_mean_pool as _gmean_pool, global_max_pool as _gmax_pool
from typing import Literal as _Lit
from pathlib import Path as _Path

from .blocks.sageblock import SageBlock as _SageBlock
from .map.mapencoder import MapEncoder as _MapEncoder
from .map.mapattention import MapSpatialAttention as _MapSpatialAttention
from .map.mapInputNorm import MapZscoreNorm as _MapZscoreNorm

class GruSage(_nn.Module):
    def __init__(self, dynamic_features_num:int, frames_num:int, gru_hidden_size:int, gru_num_layers:int, fc1dims:list[int], sage_hidden_dims:list[int]=[128, 128], fc2dims:list[int]=[50,50], out_dim:int=1, num_st_types:int=256, emb_dim:int=12, dropout:float|None=None, negative_slope:float|None=None, global_pooling:_Lit['mean', 'max','double']='double',map_included:bool=True,*, map_tensors:dict|None=None, mapenc_sage_hdims:list[int]=[8,8], mapenc_lane_embdim:int=2, map_attention_topk:int=5, map_embeddings:_torch.Tensor|None=None):
        super().__init__()
        # in order to initialize properly, if map_included is True, either map_tensors or map_embeddings must be provided, and map_attention_topk is required in any case
        if map_included:
            assert (map_tensors is not None) or (map_embeddings is not None), "If map_included is True, either map_tensors or map_embeddings must be provided"
            assert map_attention_topk is not None, "If map_included is True, map_attention_topk must be provided"
            assert (map_tensors is None) or (map_embeddings is None), "Provide either map_tensors or map_embeddings, not both"

        # create config dict for snapshot saving
        self.config_dict = {
            "dynamic_features_num": dynamic_features_num,
            "frames_num": frames_num,
            "gru_hidden_size": gru_hidden_size,
            "gru_num_layers": gru_num_layers,
            "fc1dims": fc1dims,
            "sage_hidden_dims": sage_hidden_dims,
            "fc2dims": fc2dims,
            "out_dim": out_dim,
            "num_st_types": num_st_types,
            "emb_dim": emb_dim,
            "dropout": dropout,
            "negative_slope": negative_slope,
            "global_pooling": global_pooling,
            "map_included": map_included,
            "map_attention_topk": map_attention_topk,
            "map_embeddings": map_embeddings
            # map tensors not needed as inputs -> loaded from state dict directly
        }


        #TODO validate inputs
        assert len(sage_hidden_dims) >= 1, "sage_hidden_dims must contain at least one element"
        # ...

        # 1 - embedding for station types
        self.st_emb = _nn.Embedding(num_st_types, emb_dim)

        # 2 - GRU layer to process dynamic features
        self.gru = _nn.GRU(
            input_size=dynamic_features_num,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )

        # 3. concat all input features
        last_step_dims = gru_hidden_size + 2 + emb_dim

        # 4 - fully connected layers before GraphSAGE
        ldims1 = [last_step_dims] + fc1dims
        self.fc1s = _nn.ModuleList([
            _nn.Sequential(
                _nn.Linear(ldims1[i], ldims1[i+1]),
                _nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else _nn.ReLU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(ldims1)-1)
        ])
        last_step_dims = ldims1[-1]

        # 4b add map tensors if provided
        if map_included:
            self.map_provided = True
            self.map_tensors = (map_tensors is not None)
            if map_tensors is not None:
                # input map tensors are provided, so initialize map encoder and attention modules normally
                #TODO add embedding and hidden layers specs for map encoder
                self.map_encoder = _MapEncoder(
                    map_float_features=_MapZscoreNorm.onfly(map_tensors['float_features']),
                    map_bool_features = map_tensors['bool_features'],
                    lane_type_cats=map_tensors['lane_type_cats'],
                    graph_edge_indexes=map_tensors['mgraph_edge_indexes'],
                    lane_embed_dim = mapenc_lane_embdim,
                    sage_hidden_dims=mapenc_sage_hdims,
                    dropout=dropout,
                    negative_slope=negative_slope
                )
                #TODO add attention specs
                self.map_attention = _MapSpatialAttention(
                    map_centroids=map_tensors['mseg_centroids'],
                    k_neighbors=map_attention_topk
                )
                last_step_dims += self.map_encoder.out_dim # add attentioned map embeddings to input features
            else:
                # directly provide map embeddings as buffer, so not need to initialize map encoder, only attention module
                self.register_buffer('map_embeddings', map_embeddings)
                self.map_attention = _MapSpatialAttention(
                    map_centroids=None, # do not register anything
                    k_neighbors=map_attention_topk
                )
                last_step_dims += map_embeddings.shape[1] # add attentioned map embeddings to input features
            

        # 5 - GraphSAGE layers
        sagedims = [last_step_dims] + sage_hidden_dims
        self.sage = _SageBlock(sagedims, dropout=dropout, negative_slope=negative_slope)
        last_step_dims = sagedims[-1]

        # 6 - global pooling
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


        # 7 - fully connected layers after GraphSAGE
        ldims = [last_step_dims] + fc2dims
        self.fc2s = _nn.ModuleList([
            *[_nn.Sequential(
                _nn.Linear(ldims[i], ldims[i+1]),
                _nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else _nn.ReLU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(ldims)-1)]
        ])

        # 8 - final output layer
        self.linout = _nn.Linear(ldims[-1], out_dim)

        self.dropout = dropout
        self.negative_slope = negative_slope

    @classmethod
    def from_snapshot(cls,path:_Path):
        snapshot = _torch.load(path.resolve())
        model = cls(**snapshot['config'])
        model.load_state_dict(snapshot['state_dict'])
        return model

    def save_snapshot(self, path:_Path):
        state_dict = {k:v for k,v in self.state_dict().items() if not k.startswith('map_encoder')}
        # map encoder is excluded as we directly load map tensors
        # map attention module is included instead
        with _torch.no_grad():
            cfg = self.config_dict.copy()
            cfg['map_embeddings'] = self.map_encoder() if self.map_provided else None,

        _torch.save({
            "config": cfg,
            "state_dict": state_dict,
        },path.resolve())


    def forward(self, data):
        x, edge_index, edge_attr, xsttype, batch = data.x, data.edge_index, data.edge_attr, data.xsttype, data.batch

        # 1 - embedding for station types
        st_embedded:_torch.Tensor = self.st_emb(xsttype)

        # 2 - process dynamic features with GRU
        # x shape: [batch_vnum, framenum, feature_dim]
        gru_out, hlast = self.gru(x)
        x = hlast[-1,:,:] # take last hidden state

        # 3 - concat all input features
        xdims = data.xdims
        x = _torch.cat([x, xdims,st_embedded], dim=1)

        # 4 - fc layers before GraphSAGE
        for fc in self.fc1s:
            x = fc(x)

        # 4b - map encoding and attention if map tensors provided
        if self.map_provided:
            last_pos_raw = data.pos_raw[:,-1,:]
            map_embeddings = self.map_encoder() if self.map_tensors else self.map_embeddings
            map_context = self.map_attention(
                vehicle_last_positions = last_pos_raw, # coords before zscore norm - [BATCH_SIZE, 2]
                map_embeddings = map_embeddings # [NUM_TOTAL_SEGMENTS, EMBED_DIM]
            )
            x = _torch.cat([x, map_context], dim=1)

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

        def addGradsToDict(module:_nn.Module, name:str):
            buf = [param.grad.view(-1) for pname, param in module.named_parameters() if (param.requires_grad and param.grad is not None)]
            per_layer_grads[name] = _torch.cat(buf) if len(buf) > 0 else None

        addGradsToDict(self.st_emb, 'StType Embedding')
        addGradsToDict(self.gru, 'GRU Layer')
        addGradsToDict(self.fc1s, 'FC Layers before SAGE')
        addGradsToDict(self.sage, 'GraphSAGE Layers')
        addGradsToDict(self.fc2s, 'FC Layers after SAGE')
        addGradsToDict(self.linout, 'Final Output Layer')

        # concat all grads
        tot_grads = _torch.cat([g for g in per_layer_grads.values() if g is not None]) if len(per_layer_grads) > 0 else None

        tot_norm = tot_grads.norm().item() if tot_grads is not None else None
        layer_norms ={
            name: (gr.norm().item() if gr is not None else None)
            for name, gr in per_layer_grads.items()
        }
        return tot_norm, layer_norms