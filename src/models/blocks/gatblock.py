import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import GATv2Conv as _GATv2Conv

class GATv2Block(_nn.Module):
    def __init__(self, hdims:list[int], gat_nheads:int, gat_concat:bool, *, gat_edge_fnum:int|None=None, dropout:float|None=None, negative_slope:float=0.2):
        super().__init__()

        mult = gat_nheads if gat_concat else 1

        self.convs = _nn.ModuleList([
            _GATv2Conv(
                in_channels=hdims[i]*(mult if i>0 else 1),
                out_channels=hdims[i+1],
                heads=gat_nheads,
                concat=gat_concat,
                negative_slope=negative_slope if negative_slope is not None else 0.2,
                edge_dim=gat_edge_fnum,
                dropout=dropout
            ) for i in range(len(hdims)-1)
        ])

        self.posts = _nn.ModuleList([
            _nn.Sequential(
                _nn.LayerNorm(hdims[i+1]*mult),
                _nn.ELU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(hdims)-1)
        ])

        self.gat_edge_fnum = gat_edge_fnum

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            if self.gat_edge_fnum is not None:
                raise ValueError("Expected edge_attr is required but got None")
            
            for conv, post in zip(self.convs, self.posts):
                x = conv(x, edge_index)
                x = post(x)
            return x
        else:
            if self.gat_edge_fnum is None:
                raise ValueError("Unexpected edge_attr provided when gat_edge_fnum is None")
            
            for conv, post in zip(self.convs, self.posts):
                x = conv(x, edge_index, edge_attr)
                x = post(x)
            return x
