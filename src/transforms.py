import torch as _tch
from torch_geometric.data import Data as _GData

from .utils import MetaData as _MD, FmaskType as _FMType

# define useful transformations for dataset

class AddNoise:
    def __init__(self,target:_FMType, std:float, metadata:_MD):
        self.std = std
        self.mask = metadata.getDataMask(target)
        self.flattenedTime = metadata.flattened_time
    
    def __call__(self, data:_GData)->_GData:
        # TODO:CHECK this implementation
        if self.flattenedTime:
            data.x[:,self.mask] += _tch.randn_like(data.x[:,self.mask],device=data.x.device) * self.std
        else:
            data.x[:,:,self.mask] += _tch.randn_like(data.x[:,:,self.mask],device=data.x.device) * self.std
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
        self.flattenedTime = metadata.flattened_time
        self.num_frames = metadata.frames_num
        self.posMask = metadata.getDataMask('pos')
        self.hMask = metadata.getDataMask('heading')
        self.hSinMask = metadata.getDataMask('hsin')
        self.hCosMask = metadata.getDataMask('hcos')
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

        # TODO:CHECK: why in this implementation masks are not required w.r.t. other transforms?
        if self.flattenedTime:
            # extract positions (still temporal sequence)
            positions = data.x[:,self.posMask]

            # rotate positions
            for i in range(self.num_frames):
                coords = positions[:,i*2:(i*2)+2]  # shape [num_nodes, 2]
                rotcoords = coords @ RotMat.T  # shape [num_nodes, 2]
                positions[:,i*2:(i*2)+2] = rotcoords
            data.x[:,self.posMask] = positions
        else:
            # extract positions
            positions = data.x[:,:,self.posMask]
            # rotate positions
            positions = positions @ RotMat.T  # shape [num_nodes, num_features, 2]
            data.x[:,:,self.posMask] = positions

        # adjust headings
        if self.headingEncoded:
            if self.flattenedTime:
                hsin = data.x[:,self.hSinMask]
                hcos = data.x[:,self.hCosMask]
                data.x[:,self.hSinMask] = hsin*cos_theta + hcos*sin_theta
                data.x[:,self.hCosMask] = hcos*cos_theta - hsin*sin_theta
            else:
                hsin = data.x[:,:,self.hSinMask]
                hcos = data.x[:,:,self.hCosMask]
                data.x[:,:,self.hSinMask] = hsin*cos_theta + hcos*sin_theta
                data.x[:,:,self.hCosMask] = hcos*cos_theta - hsin*sin_theta
        else:
            if self.flattenedTime:
                headings = data.x[:,self.hMask]
                data.x[:,self.hMask] = (headings + theta) % (2 * _tch.pi)
            else:
                headings = data.x[:,:,self.hMask]
                data.x[:,:,self.hMask] = (headings + theta) % (2 * _tch.pi)
        

        return data
        
        

__all__ = ['AddNoise','RemoveDimsFeatures','RandomRotate']