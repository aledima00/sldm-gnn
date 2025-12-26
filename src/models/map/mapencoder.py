import torch as _tch
import torch.nn as _nn

from ..blocks.sageblock import SageBlock as _SageBlock

class MapEncoder(_nn.Module):
    def __init__(self, map_float_features:_tch.Tensor, lane_type_cats:_tch.Tensor, graph_edge_indexes:_tch.Tensor,*,lane_embed_dim:int=2,sage_hidden_dims:list[int]=[8,8],dropout,negative_slope):
        super().__init__()
        self.register_buffer('map_float_features', map_float_features)
        self.register_buffer('lane_type_cats', lane_type_cats)
        self.register_buffer('graph_edge_indexes', graph_edge_indexes)
        self.lane_embed_dim = lane_embed_dim

        self.lane_embedding = _nn.Embedding(
            num_embeddings = int(_tch.max(lane_type_cats).item()) + 1,
            embedding_dim = lane_embed_dim
        )
        input_dim = map_float_features.shape[1] + lane_embed_dim

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
        lane_embeds = self.lane_embedding(self.lane_type_cats)
        x = _tch.cat([self.map_float_features, lane_embeds], dim=1)

        # Pass through GraphSAGE
        out = self.sage(x, self.graph_edge_indexes)

        return out