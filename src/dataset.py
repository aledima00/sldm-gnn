import torch as _tch
from torch_geometric.data import Data as _GData, Dataset as _GDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
from pathlib import Path as _Path
from torch.serialization import safe_globals
from typing import Literal as _Lit
from tqdm.auto import tqdm as _tqdm

def _statsToMuSigma(sum_x:_tch.Tensor, sum_x2:_tch.Tensor, tot_cnt:int, frames_num:int=20)->tuple[_tch.Tensor,_tch.Tensor]:

    # static features
    s_wl = sum_x[7*frames_num:]
    s_wl2 = sum_x2[7*frames_num:]
    mu_wl = s_wl / tot_cnt
    sigma_wl = ((s_wl2 / tot_cnt) - (mu_wl ** 2)).sqrt()
    
    # temporal features
    # sxt = sum_x[:7] + sum_x2[7:14] + ...
    sxt = _tch.zeros(7, device=sum_x.device)
    sxt2 = _tch.zeros(7, device=sum_x.device)
    for i in range(frames_num):
        sxt += sum_x[i*7:(i+1)*7]
        sxt2 += sum_x2[i*7:(i+1)*7]

    tcnt = tot_cnt * frames_num
    mu_t = sxt / tcnt
    sigma_t = ((sxt2 / tcnt) - (mu_t ** 2)).sqrt()

    # append static features
    mu = _tch.cat([mu_t.repeat(frames_num), mu_wl])
    sigma = _tch.cat([sigma_t.repeat(frames_num), sigma_wl])
    return mu, sigma

class MapGraph(_GDataset):
    pos_rescaling_opt_type = _Lit['none', 'center']
    def __init__(self, graphs_dirpath:_Path,*,frames_num:int=20, active_labels:list[int]=None,device:str='cpu',transform=None,normalizeZScore:bool=True):
        super().__init__(transform=transform)
        self.frames_num = frames_num
        self.dirpath = graphs_dirpath.resolve()
        self.active_labels = active_labels

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
            graph.x = (graph.x - self.mu) / self.sigma
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
        sum_x = None
        sum_x2 = None
        tot_cnt = 0
        with self.usingRawData:
            for i in _tqdm(range(self.len()), desc="Computing dataset mean and std"):
                g = self.innerGet(i)
                if sum_x is None:
                    sum_x = _tch.zeros(g.x.size(1), device=g.x.device)
                    sum_x2 = _tch.zeros(g.x.size(1), device=g.x.device)
                sum_x += g.x.sum(dim=0)
                sum_x2 += (g.x ** 2).sum(dim=0)
                tot_cnt += g.x.size(0)
        return _statsToMuSigma(sum_x, sum_x2, tot_cnt, self.frames_num)