import torch.nn as _nn
from torch_geometric.nn import SAGEConv as _SageConv

class SageBlock(_nn.Module):
    def __init__(self, hdims:list[int], dropout:float|None=None, negative_slope:float|None=None):
        super().__init__()
        assert len(hdims) >= 1, "hdims must contain at least one element"
        self.convs = _nn.ModuleList([_SageConv(hdims[i], hdims[i+1]) for i in range(len(hdims)-1)])
        self.posts = _nn.ModuleList([
            _nn.Sequential(
                _nn.LayerNorm(hdims[i+1]),
                _nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else _nn.ReLU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(hdims)-1)
        ])
    def forward(self, x, edge_index):
        for conv, post in zip(self.convs, self.posts):
            x = conv(x, edge_index)
            x = post(x)
        return x