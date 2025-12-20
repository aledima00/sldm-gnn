import torch as _tch
import torch.nn as _nn

from ..blocks.sageblock import SageBlock as _SageBlock

class MapEncoder(_nn.Module):
    def __init__(self, map_float_features:_tch.Tensor, map_edge_ids:_tch.Tensor, map_edge_indexes:_tch.Tensor,*,eid_embed_dim:int=10,sage_hidden_dims:list[int]=[8,8],dropout,negative_slope):
        super().__init__()
        self.register_buffer('map_float_features', map_float_features)
        self.register_buffer('map_edge_ids', map_edge_ids)
        self.register_buffer('map_edge_indexes', map_edge_indexes)
        self.edgeid_embed_dim = eid_embed_dim

        self.eid_embedding = _nn.Embedding(
            num_embeddings = int(_tch.max(map_edge_ids).item()) + 1,
            embedding_dim = eid_embed_dim
        )
        input_dim = map_float_features.shape[1] + eid_embed_dim

        self.sage = _SageBlock(
            [input_dim] + sage_hidden_dims,
            dropout=dropout,
            negative_slope=negative_slope
        )
        self._out_dim = sage_hidden_dims[-1]

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self):
        # Prepare input features
        edge_embeds = self.eid_embedding(self.map_edge_ids)
        x = _tch.cat([self.map_float_features, edge_embeds], dim=1)

        # Pass through GraphSAGE
        out = self.sage(x, self.map_edge_indexes)

        return out