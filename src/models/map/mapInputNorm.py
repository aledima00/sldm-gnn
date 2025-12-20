import torch as _tch

class MapZscoreNorm:
    """
    Normalizes the map input features using pre-defined mean and standard deviation.
    Args:
        map_float_features (torch.Tensor): The raw map float features tensor of shape [NUM_SEGMENTS, NUM_FEATURES].
    Returns:
        torch.Tensor: The normalized map float features tensor (same shape as input).
    """

    def __init__(self,map_float_features:_tch.Tensor):
        # zscore norm over segments as samples
        self.mu = _tch.sum(map_float_features, dim=0, keepdim=True) / map_float_features.shape[0]
        self.sigma = _tch.sqrt(_tch.sum((map_float_features - self.mu) ** 2, dim=0, keepdim=True) / map_float_features.shape[0]).clamp(min=1e-8) # TODO set specific clamping value, now empirical
    
    def __call__(self, map_input:_tch.Tensor):
        return ((map_input - self.mu) / self.sigma)
    
    @classmethod
    def onfly(cls, map_float_features:_tch.Tensor):
        normalizer = cls(map_float_features)
        return normalizer(map_float_features)    