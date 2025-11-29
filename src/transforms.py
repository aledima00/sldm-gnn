import torch as _tch
from torch_geometric.data import Data as _GData

# define useful transformations for dataset

# x features: [*[Xt, Yt, Speedt, HeadingT, PresenceFlagt, tsin, tcos]*20, width, length]
xmask = [True,False,False,False,False,False,False]*20 + [False, False]
ymask = [False,True,False,False,False,False,False]*20 + [False, False]
speed_mask = [False,False,True,False,False,False,False]*20 + [False, False]
heading_mask = [False,False,False,True,False,False,False]*20 + [False, False]
wmask = [False]*7*20 + [True, False]
lmask = [False]*7*20 + [False, True]
tmask = [True]*7*20 + [False, False]

class PosNoise:
    def __init__(self, std:float, device:str='cpu'):
        self.std = std
        self.device = device
    
    def __call__(self, data:_GData)->_GData:
        data.x[:,xmask] += _tch.randn_like(data.x[:,xmask],device=self.device) * self.std
        data.x[:,ymask] += _tch.randn_like(data.x[:,ymask],device=self.device) * self.std
        return data
    
class SpeedNoise:
    def __init__(self, std:float, device:str='cpu'):
        self.std = std
        self.device = device
    def __call__(self, data:_GData)->_GData:
        data.x[:,speed_mask] += _tch.randn_like(data.x[:,speed_mask], device=self.device) * self.std
        return data

class HeadingNoise:
    def __init__(self, std:float, device:str='cpu'):
        self.std = std
        self.device = device
    def __call__(self, data:_GData)->_GData:
        data.x[:,heading_mask] += _tch.randn_like(data.x[:,heading_mask], device=self.device) * self.std
        return data
    
class RemoveDimsFeatures:
    def __call__(self, data:_GData)->_GData:
        data.x = data.x[:, tmask]
        return data
    
class RescalePosToVCenter:
    def __init__(self):
        pass
    def __call__(self, data:_GData)->_GData:
        # rescale X,Y positions so that (0,0) is the center of the vehicle

        # extract dims
        l = data.x[:,lmask]
        h = data.x[:,heading_mask]
        x = data.x[:,xmask]
        y = data.x[:,ymask]

        # rescale
        x = x - (l/2) * _tch.cos(h)
        y = y - (l/2) * _tch.sin(h)

        # save
        data.x[:,xmask] = x
        data.x[:,ymask] = y
        return data
    
class RandomRotate:
    def __init__(self,num_frames:int=20,device:str='cpu'):
        self.num_frames = num_frames
        self.device = device
    def __call__(self, data:_GData)->_GData:
        # random rotation angle
        theta = _tch.rand(1).item() * 2 * _tch.pi
        cos_theta = _tch.cos(_tch.tensor(theta, device=self.device))
        sin_theta = _tch.sin(_tch.tensor(theta, device=self.device))

        # extract positions (still temporal sequence)
        xs = data.x[:,xmask]
        ys = data.x[:,ymask]

        # rotation matrix
        RotMat = _tch.tensor([
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta]
        ], device=self.device)

        # rotate positions
        for i in range(self.num_frames):
            coords = _tch.stack([xs[:,i], ys[:,i]], dim=1)  # shape [num_nodes, 2]
            rotated_coords = coords @ RotMat.T  # shape [num_nodes, 2]
            xs[:,i] = rotated_coords[:,0]
            ys[:,i] = rotated_coords[:,1]
        data.x[:,xmask] = xs
        data.x[:,ymask] = ys

        # adjust headings
        headings = data.x[:,heading_mask]
        headings = (headings + theta) % (2 * _tch.pi)
        data.x[:,heading_mask] = headings

        return data
        
        

