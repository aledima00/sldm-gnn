import torch as _tch
from torch_geometric.data import Data as _GData, Dataset as _GDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from pathlib import Path as _Path
from torch.serialization import safe_globals
from typing import Literal as _Lit
from tqdm.auto import tqdm as _tqdm

from .labels import LabelsEnum as _LE
from .utils import MetaData as _MD


class MapGraph(_GDataset):
    pos_rescaling_opt_type = _Lit['none', 'center']
    def __init__(self, graphs_dirpath:_Path,*, device:str='cpu',transform=None,normalizeZScore:bool=True,metadata:_MD=None,zscore_mu_sigma:tuple[dict,dict]|None=None):
        super().__init__(transform=transform)
        if metadata is None:
            metadata = _MD.loadJson(graphs_dirpath / 'metadata.json')
        
        self.frames_num = metadata.frames_num
        self.hasDims = metadata.has_dims
        self.active_labels = set(metadata.active_labels)
        self.heading_encoded = metadata.heading_encoded
        self.n_temp_feats = metadata.n_node_temporal_features
        self.flatten_time_as_graphs = metadata.flatten_time_as_graphs
        if self.active_labels is None:
            self.active_labels = set([l.value for l in _LE])

        self.dirpath = graphs_dirpath.resolve()

        # list of .pt graph files
        self.paths = list(sorted(self.dirpath.glob("*.pt")))
        self.device = device

        self.normalizeZScore = normalizeZScore
        if normalizeZScore:
            if zscore_mu_sigma is not None:
                self.mu, self.sigma = zscore_mu_sigma
            else:
                self.mu, self.sigma = self.computeMuSigma()
    
    def getMuSigma(self):
        if not hasattr(self, 'mu') or not hasattr(self, 'sigma') or self.mu is None or self.sigma is None:
            mu_sigma= self.computeMuSigma()
            self.mu, self.sigma = mu_sigma
        return self.mu, self.sigma

    def len(self):
        return len(self.paths)
    
    def get(self, idx):
        return self.innerGet(idx)
    
    def _mapLabelToDenseTensor(self, y:_tch.Tensor):
        #TODO:CHECK this implementation for possible optimizations
        if max(self.active_labels) >= y.size(0) or min(self.active_labels) < 0:
            raise ValueError(f"Active labels contain invalid label indices for y of size {y.size(0)}: {self.active_labels}")
        if len(self.active_labels) < y.size(0):
            # adjust y to only active labels
            new_y = _tch.zeros((len(self.active_labels)), dtype=y.dtype)
            for i,c in enumerate(self.active_labels):
                new_y[i] = y[c]
            return new_y
        else:
            return y

    def getRawByPid(self,pid:int):
        fname = (self.dirpath / f"pack_{pid}.pt").resolve()
        if not fname.exists():
            raise FileNotFoundError(f"Graph file for pack id {pid} not found at path: {fname}")
        with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            graph:_GData = _tch.load(fname, map_location=self.device)
            if graph.y is not None and self.active_labels is not None:
                graph.y = self._mapLabelToDenseTensor(graph.y)
        return graph

    def innerGet(self,idx):
        with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
            graph:_GData = _tch.load(self.paths[idx], map_location=self.device)
            if graph.y is not None and self.active_labels is not None:
                graph.y = self._mapLabelToDenseTensor(graph.y)
        if self.transform:
            graph = self.transform(graph)
        if self.normalizeZScore:
            if self.flatten_time_as_graphs:
                # presence mask is already removed in this mode
                graph.pos_raw = graph.x[:, :2].clone()  # store raw positions before normalization
                graph.x = (graph.x - self.mu["x"]) / self.sigma["x"]
            else:
                # z-score normalization on features from 0 to -1 (excluding presence mask)
                graph.pos_raw = graph.x[:,:, :2].clone()  # store raw positions before normalization
                graph.x[:,:,:-1] = (graph.x[:,:,:-1] - self.mu["x"]) / self.sigma["x"]
            if self.hasDims:
                graph.xdims = (graph.xdims - self.mu["xdims"]) / self.sigma["xdims"]

        #TODO - add flattening option here in case is needed, only as postproc
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
    
    def computeMuSigma(self):
        nfeats = (self.n_temp_feats - (1 if not self.flatten_time_as_graphs else 0))
        # x,y,speed, heading(encoded?), (time enc?) 
        # presence mask not included in zscore stats compute, used as mask also there
        
        # reduce over vehicles and time
        rshp = (1,1,nfeats) if not self.flatten_time_as_graphs else (1,nfeats)
        sum_x = _tch.zeros(rshp,device=self.device,dtype=_tch.float)
        sum_x2 = _tch.zeros(rshp,device=self.device,dtype=_tch.float)
        
        if self.hasDims:
            sum_xdims = _tch.zeros((1,2), device=self.device)
            sum_xdims2 = _tch.zeros((1,2), device=self.device)

        tot_cnt = 0
        vcnt = 0

        with self.usingRawData:
            if not self.flatten_time_as_graphs:
                for i in _tqdm(range(self.len()), desc="Computing dataset mean and std"):
                    g = self.innerGet(i) # get raw graph

                    for vi in range(g.x.size(0)):
                        # sum faetures where presence mask is true
                        gv = g.x[vi:vi+1,:,:]
                        pmask = gv[0,:, -1] > 0.5  # presence mask
                        gv = gv[:,pmask,:-1] # exclude presence mask from stats

                        # sum over time
                        sum_x += gv.sum(dim=1,keepdim=True)
                        sum_x2 += (gv ** 2).sum(dim=1,keepdim=True)
                        tot_cnt += pmask.sum().item() # count of frames for this vehicle
                    
                    if self.hasDims:
                        xdims = g.xdims
                        sum_xdims += xdims.sum(dim=0,keepdim=True)
                        sum_xdims2 += (xdims ** 2).sum(dim=0,keepdim=True)
                        vcnt += g.xdims.size(0)
            else:
                for i in _tqdm(range(self.len()), desc="Computing dataset mean and std"):
                    g = self.innerGet(i) # get raw graph
                    
                    # time already flattened as different graphs, so sum directly only over nodes
                    gv = g.x
                    sum_x += gv.sum(dim=0,keepdim=True)
                    sum_x2 += (gv ** 2).sum(dim=0,keepdim=True)
                    tot_cnt += gv.size(0) # count of nodes and frames together

                    if self.hasDims:
                        xdims = g.xdims
                        sum_xdims += xdims.sum(dim=0,keepdim=True)
                        sum_xdims2 += (xdims ** 2).sum(dim=0,keepdim=True)
                vcnt = tot_cnt  # in this mode, consider vcnt same as tot_cnt for dims stats
                    

        if self.hasDims:
            mu_xdims = sum_xdims / vcnt
            sigma_xdims = ((sum_xdims2 / vcnt) - (mu_xdims ** 2)).sqrt()

        mu_x = sum_x / tot_cnt
        sigma_x = ((sum_x2 / tot_cnt) - (mu_x ** 2)).sqrt().clamp(min=1e-8) # TODO set specific clamping value, now empirical
        return ({"x": mu_x, "xdims": mu_xdims if self.hasDims else None}, {"x": sigma_x, "xdims": sigma_xdims if self.hasDims else None})
