import torch as _tch
import torch.nn as _nn
import torch.nn.functional as _F

class MapSpatialAttention(_nn.Module):
    def __init__(self, map_centroids:_tch.Tensor|None, k_neighbors=5):
        super().__init__()
        # map_centroids: [NUM_TOTAL_SEGMENTS, 2] -> coordinates of segment centroids
        # map_embeddings: [NUM_TOTAL_SEGMENTS, EMBED_DIM] -> lane embeddings (after mapGNN)
        if map_centroids is not None:
            self.register_buffer('map_centroids', map_centroids, persistent=True) # persistent allow to save in state_dict but not update during training
        self.k = k_neighbors # pick knn segments for attention
        
        # attention scores based on euclidian distance rather than direct transformer dot-product, as we want to focus on spatial neighborhood
        # small MLP to compute function of distance -> attention score
        self.attn_mlp = _nn.Sequential(
            _nn.Linear(1, 16), # Input: scalar distance between vehicle and segment
            _nn.ReLU(),
            _nn.Linear(16, 1)  # Output: REAL attention score
        )

    def forward(self, vehicle_last_positions, map_embeddings):
        """
        vehicle_last_positions: [BATCH_SIZE, 2] -> (x, y) current vehicle positions
        segments: [NUM_TOTAL_SEGMENTS, N_SEGMENT_FEATURES]
        Returns:
            context_vector: [BATCH_SIZE, N_SEGMENT_FEATURES] -> aggregated map context for each vehicle
        """
        
        # 1. Compute Distances (Pairwise Distance)
        # Expand dimensions for broadcasting:
        # Vehicles: [BS, <newaxis on N_SEG_F>, 2] - Map Centroids: [<newaxis on BS>, N_SEG_F, 2] -> Result [BS, N_SEG_F, 2] -> Norm -> [BS, N_SEG_F]
        #print(f"dimensions:\n - vehicle_last_positions: {vehicle_last_positions.shape}\n - map_centroids: {self.map_centroids.shape}\n - map_embeddings: {map_embeddings.shape}")
        diff = vehicle_last_positions.unsqueeze(1) - self.map_centroids.unsqueeze(0)
        dists = _tch.norm(diff, dim=2)
        # Euclidean Distance [BS, N_SEG_F]
        
        # 2. Select KNNs
        # val_dists: [BS, K], indices: [BS, K]
        neg_dists, indices = _tch.topk(-dists, k=self.k, dim=1) # negative to use topk for smallest distances
        k_dists = -neg_dists 
        
        # 3. Retrieve Corresponding Map Embeddings
        # [BS, K, N_SEG_F]
        # Expand map_embeddings to allow batch indexing
        batch_map_embeds = map_embeddings[indices, :] 
        
        # 4. Compute Attention Weights (Softmax on distance)
        # The closer (smaller distance), the higher the weight. The function is learned via MLP.
        # k_dists: [BS, K] -> reshape to [BS, K, 1] for MLP
        attn_scores = self.attn_mlp(k_dists.unsqueeze(2)).squeeze(2) # [BS, K]
        weights = _F.softmax(attn_scores, dim=1).unsqueeze(2) # [BS, K, 1]
        
        # 5. Weighted Aggregation
        # Weighted sum of the K embeddings: [BS, N_SEG_F]
        context_vector = _tch.sum(batch_map_embeds * weights, dim=1)
        
        return context_vector