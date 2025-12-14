import torch as _tch
from torch_geometric.data import Data as _GData

from .utils import MetaData as _MD, FmaskType as _FMType

# define useful transformations for dataset
#TODO: implement masks that scales in dimensions, in order to use [:,mask] indexing also when [:, :, mask] would be needed

class AddNoise:
    def __init__(self,target:_FMType, std:float, metadata:_MD, prop_to_speed:bool=False):
        self._std = std
        # std assumed nominal std if prop_to_speed is False or max std if prop_to_speed is True
        self.mask = metadata.getFeaturesMask(target)
        self.speedMask = metadata.getFeaturesMask('speed')
        self.flattened_time_as_graphs = metadata.flatten_time_as_graphs
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
            speed = data.x[:,self.speedMask] if self.flattened_time_as_graphs else data.x[:,:,self.speedMask]
            std = self.getStd(speed=speed)
        else:
            std = self.getStd()

        
        if self.flattened_time_as_graphs:
            data.x[:,self.mask] += _tch.randn_like(data.x[:,self.mask],device=data.x.device) * std
        else:
            data.x[:,:,self.mask] += _tch.randn_like(data.x[:,:,self.mask],device=data.x.device) * std
        return data
    
class RemoveDimsFeatures:
    def __init__(self,metadata:_MD):
        self.hasDims = metadata.has_dims
        
    def __call__(self, data:_GData)->_GData:
        if self.hasDims:
            # remove width and length from static features (first 2 features)
            del data.xdims
            # TODO:CHECK: is this necessary or should i just ignore it, avoiding in the model the use of data.xdims directly?
        return data
    
class RandomRotate:
    def __init__(self, metadata:_MD):
        self.num_frames = metadata.frames_num
        self.posMask = metadata.getFeaturesMask('pos')
        if metadata.heading_encoded:
            self.hSinMask = metadata.getFeaturesMask('hsin')
            self.hCosMask = metadata.getFeaturesMask('hcos')
        else:
            self.hMask = metadata.getFeaturesMask('heading')
        self.headingEncoded = metadata.heading_encoded
    def __call__(self, data:_GData)->_GData:
        # random rotation angle
        # TODO:CHECK this implementation
        dev = data.x.device
        theta = _tch.rand(1,device=dev).item() * 2 * _tch.pi
        cos_theta = _tch.cos(_tch.tensor(theta, device=dev))
        sin_theta = _tch.sin(_tch.tensor(theta, device=dev))
        # define rotation matrix
        RotMat = _tch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], device=dev)

        # extract positions
        positions = data.x[:,:,self.posMask]
        # rotate positions
        positions = positions @ RotMat.T  # shape [num_nodes, num_features, 2]
        data.x[:,:,self.posMask] = positions

        # adjust headings
        if self.headingEncoded:
            hsin = data.x[:,:,self.hSinMask]
            hcos = data.x[:,:,self.hCosMask]
            data.x[:,:,self.hSinMask] = hsin*cos_theta + hcos*sin_theta
            data.x[:,:,self.hCosMask] = hcos*cos_theta - hsin*sin_theta
        else:
            headings = data.x[:,:,self.hMask]
            data.x[:,:,self.hMask] = (headings + theta) % (2 * _tch.pi)
        

        return data
        
        

__all__ = ['AddNoise','RemoveDimsFeatures','RandomRotate']