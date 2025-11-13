import torch as _tch
from torch_geometric.data import Data as _GData, Dataset as _GDataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, GlobalStorage
import pandas as _pd
from pathlib import Path as _Path
from tqdm.auto import tqdm as _tqdm
from torch.serialization import safe_globals

class MapGraph(_GDataset):
    def __init__(self, path:_Path, transform=None,*,frames_num:int=20,m_radius:float=10.0,labelspath:_Path=None,n_labels:int=None,tqdm_pos:int=-1,gCacheEnabled=True):
        super().__init__()
        self.transform = transform
        self.frames_num = frames_num
        self.m_radius = m_radius
        self.tqdm_pos = tqdm_pos
        self.gpath = path.parent.resolve() / 'graphs'
        self.gCacheEnabled = gCacheEnabled
        if labelspath is not None:
            self.labels_df = _pd.read_csv(labelspath, dtype={'PackId': 'uint32', 'MLBEncoded': 'uint16'},skipinitialspace=True)
            if n_labels is None:
                raise ValueError("n_labels must be specified when labelspath is provided")
            self.n_labels = n_labels
        self.df = _pd.read_parquet(path.resolve())
        self.df['VehicleId'] = self.df['VehicleId'].astype('string')
        self.df['PresenceFlag'] = 1
        self.df['PresenceFlag'] = self.df['PresenceFlag'].astype('float16')
        new_rows_df = _pd.DataFrame(columns=self.df.columns).astype(self.df.dtypes.to_dict())
        for pid, pg in self.df.groupby('PackId'):
            for vid, vg in pg.groupby('VehicleId'):
                stType = vg['stType'].iloc[0]
                vframes = vg['FrameId'].unique()
                missing_frames = set(range(self.frames_num)) - set(vframes)
                if len(missing_frames) != 0:
                    for mf in missing_frames:
                        new_row = _pd.DataFrame({
                            'VehicleId': [vid],
                            'stType': [stType],
                            'X': [0.0],
                            'Y': [0.0],
                            'Speed': [0.0],
                            'Angle': [0.0],
                            'FrameId': [mf],
                            'PackId': [pid],
                            'PresenceFlag': [0.0]
                        })
                        new_row = new_row.astype(self.df.dtypes.to_dict())
                        new_rows_df = _pd.concat([new_rows_df, new_row ], ignore_index=True)

        self.df = _pd.concat([self.df, new_rows_df], ignore_index=True)
        self.pack_ids = self.df['PackId'].unique().tolist()
    
    def __len__(self):
        return len(self.pack_ids)
    def __getitem__(self, idx):
        pack_id = self.pack_ids[idx]
        torch_save_path = (self.gpath / f"graph_{idx:04d}.pt").resolve()
        if self.gCacheEnabled and torch_save_path.exists():
            with safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
                graph = _tch.load(torch_save_path)
        else:
            graph = self.pack_to_graph(pack_id)
            if self.gCacheEnabled:
                _tch.save(graph, torch_save_path)
        return graph
    
    def pack_to_graph(self, pack_id:int):
        pack_df = self.df[self.df['PackId'] == pack_id].drop(columns=['PackId'])
        pack_df['stType'] = pack_df['stType'].astype('long')
        pack_df = pack_df.sort_values(['VehicleId','FrameId'])
        fnames = ['X','Y','Speed','Angle','PresenceFlag','stType']
        raw_feats = pack_df[fnames].to_numpy().reshape(-1, 20, 6)
        static_feats = raw_feats[:,:,:5]
        st_types_feats = _tch.tensor(raw_feats[:,0,5:], dtype=_tch.long)  # vehicle stType repeated for all frames   
        temporal_feats = static_feats.reshape(-1, 20*5)

        x = _tch.tensor(temporal_feats, dtype=_tch.float)

        # edge index construction based on distance of trajectories
        ## min distance used for threshold
        ## inverse of avg distance used for edge value

        coords = pack_df[['VehicleId','FrameId','X','Y']]
        vehicle_ids = coords['VehicleId'].unique()
        edge_index_list = []
        edge_attr_list = []
        for i, vid1 in _tqdm(enumerate(vehicle_ids), total=len(vehicle_ids), desc=f"Building edges for PackId {pack_id}",position=self.tqdm_pos if self.tqdm_pos >= 0 else 0, leave=False, disable=self.tqdm_pos < 0):
            traj1 = _tch.tensor(coords[coords['VehicleId'] == vid1].sort_values('FrameId')[['X','Y']].to_numpy())
            for j, vid2 in enumerate(vehicle_ids):
                if i != j:
                    traj2 = _tch.tensor(coords[coords['VehicleId'] == vid2].sort_values('FrameId')[['X','Y']].to_numpy())
                    dists = _tch.norm(traj1 - traj2, dim=1)
                    min_dist = _tch.min(dists).item()
                    if min_dist <= self.m_radius:
                        avg_dist = _tch.mean(dists).item()
                        edge_index_list.append([i,j])
                        edge_attr_list.append([1.0 / (avg_dist + 1e-6)])  # avoid div by zero

        edge_index = _tch.tensor(edge_index_list, dtype=_tch.long).t().contiguous()
        edge_attr = _tch.tensor(edge_attr_list, dtype=_tch.float)

        if not hasattr(self, 'labels_df'):
            return _GData(x=x, st_types_feats=st_types_feats, edge_index=edge_index, edge_attr=edge_attr)
        else:
            # finally, if labels are available, expand them as graph-level multi-label
            labels_row = self.labels_df[self.labels_df['PackId'] == pack_id]
            if labels_row.empty:
                raise ValueError(f"No labels found for PackId {pack_id}")
            labels = labels_row.iloc[0]['MLBEncoded']
            # labels are stored as bitmask in an integer
            y = _tch.zeros((self.n_labels,), dtype=_tch.float)
            for c in range(self.n_labels):
                if (labels >> c) & 1:
                    y[c] = 1.0 
            # return graph data object
            return _GData(x=x, st_types_feats=st_types_feats, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
    def save(self,*,tqdm_pos:int=-1):
        """
        Loops through the dataset and saves pre-computed graphs to the specified path.
        """
        path = self.gpath.resolve()
        path.mkdir(parents=True, exist_ok=True)
        for i in _tqdm(range(len(self)), desc="Saving graphs", position=tqdm_pos if tqdm_pos >= 0 else 0, leave=True, disable=tqdm_pos < 0):
            torch_save_path = (path / f"graph_{i:04d}.pt").resolve()
            if not torch_save_path.exists():
                # skip already existing graphs
                graph = self[i]
                _tch.save(graph, torch_save_path)