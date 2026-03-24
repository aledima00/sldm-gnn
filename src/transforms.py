import torch as _tch
from torch_geometric.data import Data as _GData

from .utils import MetaData as _MD, FmaskType as _FMType

# define useful transformations for dataset

class AddNoise:
    def __init__(self,target:_FMType, std:float, metadata:_MD, prop_to_speed:bool=False):
        self._std = std
        # std assumed nominal std if prop_to_speed is False or max std if prop_to_speed is True
        self.mask = metadata.getFeaturesMask(target)
        self.speedMask = metadata.getFeaturesMask('speed')
        self.prop_to_speed = prop_to_speed        

    def getStd(self,*,speed=None):
        if self.prop_to_speed:
            frameMaxStd = self._std
            return (1-_tch.exp(-speed/10)) * frameMaxStd
        else:
            return self._std

    
    def __call__(self, data:_GData)->_GData:
        # TODO:CHECK this implementation
        if self.prop_to_speed:
            speed = data.x[:,:,self.speedMask]
            std = self.getStd(speed=speed)
        else:
            std = self.getStd()

        
        data.x[:,:,self.mask] += _tch.randn_like(data.x[:,:,self.mask],device=data.x.device) * std
        return data
    
class RemoveDimsFeatures:
    def __init__(self,metadata:_MD):
        pass
    def __call__(self, data:_GData)->_GData:
        if hasattr(data, 'xdims'):
            del data.xdims
        return data

class CutFrames:
    def __init__(self, cut:int):
        self.cut = cut

    def __call__(self, data:_GData)->_GData:
        # assume data.x shape is [num_nodes, num_frames, num_features]
        data.x = data.x[:, :self.cut, :]
        return data
        
        

__all__ = ['AddNoise','RemoveDimsFeatures','RandomRotate','CutFrames']