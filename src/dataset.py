import torch as _tch
from torch_geometric.data import Data as _GData, Dataset as _GDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from pathlib import Path as _Path
from torch.serialization import safe_globals
from typing import Literal as _Lit
from tqdm.auto import tqdm as _tqdm

from .labels import LabelsEnum as _LE
from .utils import MetaData as _MD

def _statsToMuSigma(sum_x:_tch.Tensor, sum_x2:_tch.Tensor, tot_cnt:int, frames_num:int=20, flattenedTime:bool=False)->tuple[_tch.Tensor,_tch.Tensor]:
    if flattenedTime:
        nfeats = sum_x.size(1) // frames_num
        sxt = _tch.zeros((1,nfeats), device=sum_x.device)
        sxt2 = _tch.zeros((1,nfeats), device=sum_x.device)

        for i in range(frames_num):
            sxt += sum_x[:,i*nfeats:(i+1)*nfeats]
            sxt2 += sum_x2[:,i*nfeats:(i+1)*nfeats]

        tcnt = tot_cnt * frames_num
        mu_t = sxt / tcnt
        sigma_t = ((sxt2 / tcnt) - (mu_t ** 2)).sqrt()

        # repeat for each time frame
        mu = mu_t.repeat(1,frames_num)
        sigma = sigma_t.repeat(1,frames_num)
        return mu, sigma
    else:
        # already have time dimension, so sum over it
        sxt = sum_x.sum(dim=1,keepdim=True)
        sxt2 = sum_x2.sum(dim=1,keepdim=True)
        tcnt = tot_cnt * frames_num
        mu = sxt / tcnt
        sigma = ((sxt2 / tcnt) - (mu ** 2)).sqrt()
        return mu, sigma

class MapGraph(_GDataset):
    pos_rescaling_opt_type = _Lit['none', 'center']
    def __init__(self, graphs_dirpath:_Path,*, device:str='cpu',transform=None,normalizeZScore:bool=True,metadata:_MD=None):
        super().__init__(transform=transform)
        if metadata is None:
            metadata = _MD.loadJson(graphs_dirpath / 'metadata.json')
        
        self.frames_num = metadata.frames_num
        self.hasDims = metadata.has_dims
        self.flattenedTime = metadata.flattened_time
        self.active_labels = metadata.active_labels
        self.heading_encoded = metadata.heading_encoded
        self.time_encoded = metadata.sin_cos_time_enc
        if self.active_labels is None:
            self.active_labels = [l.value for l in _LE]

        self.dirpath = graphs_dirpath.resolve()

        # list of .pt graph files
        self.paths = list(sorted(self.dirpath.glob("*.pt")))
        self.device = device

        self.normalizeZScore = normalizeZScore
        if normalizeZScore:
            self.mu, self.sigma = self.getMuSigma()
    
    def len(self):
        return len(self.paths)
    
    def get(self, idx):
        return self.innerGet(idx)

    def innerGet(self,idx):
        with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            graph:_GData = _tch.load(self.paths[idx], map_location=self.device)
            if graph.y is not None and self.active_labels is not None:
                 if max(self.active_labels) > graph.y.size(0)-1:
                    raise ValueError(f"Max Active label index {max(self.active_labels)} exceeds available labels in graph.y (TOT:{graph.y.size(0)}) for graph idx {idx}")
                 if len(self.active_labels) != graph.y.size(0):
                    # adjust y to only active labels
                    new_y = _tch.zeros((len(self.active_labels)), dtype=graph.y.dtype)
                    for i,c in enumerate(self.active_labels):
                        new_y[i] = graph.y[c]
                    graph.y = new_y
        if self.transform:
            graph = self.transform(graph)
        if self.normalizeZScore:
            # z-score normalization
            graph.x = (graph.x - self.mu["x"]) / self.sigma["x"]
            if self.hasDims:
                graph.xdims = (graph.xdims - self.mu["xdims"]) / self.sigma["xdims"]
        return graph
    
    @property
    def usingRawData(self):
        """ Returns an object with __enter__ and __exit__ methods to be used in a with statement to temporarily disable normalization and transforms. """
        class RawDataContext:
            def __init__(self, dataset:MapGraph):
                self.dataset = dataset
                self.prevNormState = dataset.normalizeZScore
                self.prevTransform = dataset.transform
            def __enter__(self):
                self.dataset.normalizeZScore = False
                self.dataset.transform = None
            def __exit__(self, exc_type, exc_value, traceback):
                self.dataset.normalizeZScore = self.prevNormState
                self.dataset.transform = self.prevTransform
        return RawDataContext(self)
    
    def getMuSigma(self):
        nfeats = 4 + (2 if self.heading_encoded else 1) + (2 if self.time_encoded else 0)
        shapex = (1,self.frames_num*nfeats) if self.flattenedTime else (1,self.frames_num,nfeats)

        sum_x = _tch.zeros(shapex,device=self.device,dtype=_tch.float)
        sum_x2 = _tch.zeros(shapex,device=self.device,dtype=_tch.float)
        
        if self.hasDims:
            sum_xdims = _tch.zeros((1,2), device=self.device)
            sum_xdims2 = _tch.zeros((1,2), device=self.device)

        vcnt = 0

        with self.usingRawData:
            for i in _tqdm(range(self.len()), desc="Computing dataset mean and std"):
                g = self.innerGet(i) # get raw graph

                sum_x += g.x.sum(dim=0,keepdim=True)
                sum_x2 += (g.x ** 2).sum(dim=0,keepdim=True)
                
                if self.hasDims:
                    sum_xdims += g.xdims.sum(dim=0,keepdim=True)
                    sum_xdims2 += (g.xdims ** 2).sum(dim=0,keepdim=True)
                vcnt += g.x.size(0)

        if self.hasDims:
            mu_xdims = sum_xdims / vcnt
            sigma_xdims = ((sum_xdims2 / vcnt) - (mu_xdims ** 2)).sqrt()
        mu_x, sigma_x = _statsToMuSigma(sum_x, sum_x2, vcnt, self.frames_num, flattenedTime=self.flattenedTime)
        return ({"x": mu_x, "xdims": mu_xdims if self.hasDims else None}, {"x": sigma_x, "xdims": sigma_xdims if self.hasDims else None})
