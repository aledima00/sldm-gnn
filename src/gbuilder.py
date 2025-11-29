import multiprocessing as _mp
from multiprocessing import Queue as _Queue
import pandas as _pd
import numpy as _np
import torch as _tch
from typing import Literal as _Lit
from torch_geometric.data import Data as _GData
from pathlib import Path as _Path
from shutil import rmtree as _rmrf
from tqdm.auto import tqdm as _tqdm
import pyarrow.parquet as _pqt
from pyarrow import RecordBatch as _Rbatch

def rescaleToCenter(features:_tch.Tensor)->_tch.Tensor:
    """
    Rescales the (X,Y) coordinates in the graph data based on the vehicle dimensions (width, length) and on the angle.
    Assuming original coordinates are taken from the center of the front border of the vehicle box,
    new coordinates will be in the center of the vehicle box.
    """
    x = features.clone()
    xs = x[:,0]
    ys = x[:,1]
    # x[:,2] is speed
    angles = x[:,3]
    # x[:,4] is PresenceFlag
    # widths = x[:, 5] # unused
    lengths = x[:, 6]
    # x[:,7] is tsin
    # x[:,8] is tcos


    # apply offsets
    x[:,0] = xs - (lengths / 2) * _tch.cos(angles)
    x[:,1] = ys - (lengths / 2) * _tch.sin(angles)

    return x

def pack2graph(*,vinfo_df:_pd.DataFrame,m_radius:float,active_labels:list[int]=None,gpath:_Path,progress_queue:_Queue,data_src_queue:_Queue)->_GData:
    
    item = data_src_queue.get()
    while item is not None:
        pack_id, pack_df, mlb = item
        pack_df = pack_df.merge(vinfo_df, on='VehicleId', how='inner')
        pack_df['stType'] = pack_df['stType'].astype('long')
        pack_df = pack_df.sort_values(['VehicleId','FrameId'])
        fnames = ['X','Y','Speed','Angle','PresenceFlag', 'tsin', 'tcos', 'width','length', 'stType']
        raw_feats = pack_df[fnames].to_numpy().reshape(-1, 20, 10)  # num_vehicles x num_frames x num_features
        
        temporal_feats = raw_feats[:,:,:7].reshape(-1, 20*7) # flatten temporal features
        dims_feats = raw_feats[:,0,7:9] # first row only, as it is a static feature
        # categorical features are treated separately (embedding)
        st_types_feats = _tch.tensor(raw_feats[:,0,9:], dtype=_tch.long) # first row only, as it is a static feature

        # x tensor: concat temporal and dim features
        x = _tch.tensor(_np.concatenate([temporal_feats, dims_feats], axis=1), dtype=_tch.float)

        # edge index construction based on distance of trajectories
        ## min distance used for threshold
        ## inverse of avg distance used for edge value

        tjsdf = pack_df[['VehicleId','FrameId','X','Y','PresenceFlag']] # df of trajectories
        vehicle_ids = tjsdf['VehicleId'].unique()
        edge_index_list = []
        edge_attr_list = []
        for i, vid1 in enumerate(vehicle_ids):
            traj1df = tjsdf[tjsdf['VehicleId'] == vid1].sort_values('FrameId')[['X','Y','PresenceFlag']]
            traj1 = traj1df[['X','Y']].to_numpy()
            for j, vid2 in enumerate(vehicle_ids):
                if i != j:
                    traj2df = tjsdf[tjsdf['VehicleId'] == vid2].sort_values('FrameId')[['X','Y','PresenceFlag']]
                    traj2 = traj2df[['X','Y']].to_numpy()

                    # compute distances
                    dists:_np.ndarray = _np.linalg.norm(traj1 - traj2, axis=1)
                    # consider only frames where both vehicles are present
                    presence_mask = (traj1df['PresenceFlag'].to_numpy() > 0.5) & (traj2df['PresenceFlag'].to_numpy() > 0.5)
                    dists[~presence_mask] = _np.inf
                    if dists.min() <= m_radius:
                        avg_dist = _np.mean(dists[_np.isfinite(dists)])
                        edge_index_list.append([i,j])
                        edge_attr_list.append([1.0 / (avg_dist + 1e-6)])  # avoid div by zero

        edge_index = _tch.tensor(edge_index_list, dtype=_tch.long).t().contiguous()
        edge_attr = _tch.tensor(edge_attr_list, dtype=_tch.float)

        if mlb is None:
            gdata = _GData(x=x, st_types_feats=st_types_feats, edge_index=edge_index, edge_attr=edge_attr)
        else:

            # labels are stored as bitmask in an integer
            y = _tch.zeros((len(active_labels),), dtype=_tch.float)
            for i,c in enumerate(active_labels):
                if mlb & (1 << c):
                    y[i] = 1.0 
            # return graph data object
            gdata = _GData(x=x, st_types_feats=st_types_feats, edge_index=edge_index, edge_attr=edge_attr, y=y)

        # print(f"gdta:{gdata}")
        # print(f"gdata types: x:{gdata.x.dtype}, edge_index:{gdata.edge_index.dtype}, edge_attr:{gdata.edge_attr.dtype}, st_types_feats:{gdata.st_types_feats.dtype}, y:{gdata.y.dtype if gdata.y is not None else 'None'}")
        _tch.save(gdata, gpath / f'pack_{pack_id}.pt')
        progress_queue.put(1)
        item = data_src_queue.get()
    return

def logworker(*,progress_queue:_Queue,total:int):
    progress = _tqdm(total=total, desc="Building Pack Graphs...", unit="graphs")
    done=0
    while done < total:
        val = progress_queue.get()
        done += val
        progress.update(val)
    progress.close()


class GraphsBuilder:
    def __init__(self,dirpath:_Path,frames_num:int=20,m_radius:float=10.0, active_labels:list[int]=None):
        self.frames_num = frames_num
        self.m_radius = m_radius

        self.dirpath = dirpath.resolve()
        self.gpath = self.dirpath / '.graphs'

        self.xpath = self.dirpath / 'packs.parquet'
        ypath = self.dirpath / 'labels.parquet'
        vpath = self.dirpath / 'vinfo.parquet'

        # load y df if available
        if ypath.exists() and ypath.is_file():
            if active_labels is None or len(active_labels) == 0:
                raise ValueError("active_labels must be specified and non-empty when labelspath is provided")
            # check positive integers
            for c in active_labels:
                if not isinstance(c, int) or c < 0:
                    raise ValueError("active_labels must contain only non-negative integers")
            self.active_labels = active_labels
            self.labels_df = _pd.read_parquet(ypath).astype({'PackId':'uint32','MLBEncoded':'uint16'})
            #print(f"Loaded labels ({len(self.labels_df)}), like:\n{self.labels_df.head(3)}\nof types:\n{self.labels_df.dtypes.to_dict()}")

        # load vinfo df if available
        if vpath.exists() and vpath.is_file():
            self.vinfo_df = _pd.read_parquet(vpath).astype({'VehicleId':'string', 'width':'float32', 'length':'float32', 'stType':'uint8'})
            # set NaN w/l to 0.0
            self.vinfo_df['width'] = self.vinfo_df['width'].fillna(0.0)
            self.vinfo_df['length'] = self.vinfo_df['length'].fillna(0.0)
            #print(f"Loaded vehicle info ({len(self.vinfo_df)}), like:\n{self.vinfo_df.head(3)}\nof types:\n{self.vinfo_df.dtypes.to_dict()}")

    def finalizepdf(self,packdf:_pd.DataFrame)->_pd.DataFrame:
        pdf = packdf.copy()
        pdf['PresenceFlag'] = 1.0
        pdf['PresenceFlag'] = pdf['PresenceFlag'].astype('float16')
        pdf.drop(columns=['PackId'], inplace=True)

        for vid, vg in pdf.groupby('VehicleId'):
            vframes = vg['FrameId'].unique()
            missing_frames = set(range(self.frames_num)) - set(vframes)
            if len(missing_frames) != 0:
                for mf in missing_frames:
                    new_row = _pd.DataFrame([{
                        'VehicleId': vid,
                        'X': 0.0,
                        'Y': 0.0,
                        'Speed': 0.0,
                        'Angle': 0.0,
                        'FrameId': mf,
                        'PresenceFlag': 0.0
                    }])
                    new_row = new_row.astype(pdf.dtypes.to_dict())
                    pdf = _pd.concat([pdf, new_row], ignore_index=True)

        # apply sin+cos temporal encoding as 2 extra features
        pdf['tsin'] = pdf['FrameId'].map(lambda x: _np.sin(2 * _np.pi * x / self.frames_num)).astype('float16')
        pdf['tcos'] = pdf['FrameId'].map(lambda x: _np.cos(2 * _np.pi * x / self.frames_num)).astype('float16')
        pdf = pdf.sort_values(['FrameId', 'VehicleId']).reset_index(drop=True)
        return pdf

    @staticmethod
    def splitPdfsByPid(pdf:_pd.DataFrame)->list[tuple[int,_pd.DataFrame]]:
        """
        Takes a batch df and splits it into a list of dataframes, one per PackId.
        """
        packs = []
        for pid, pg in pdf.groupby('PackId'):
            packs.append((pid, pg))
        return packs
    
    def mlbByPid(self,pid:int)->int|None:
        # HERE WE CAN ALSO SET SINGLE CHECK AT INIT FOR LABELS.PQT PRESENCE OR NOT
        if hasattr(self, 'labels_df'):
            row = self.labels_df[self.labels_df['PackId'] == pid]
            if len(row) > 1:
                raise ValueError(f"Multiple label rows found for PackId {pid}")
            elif len(row) == 1:
                return int(row['MLBEncoded'].values[0])
        return None

    def head(self,num)->str:
        pack_dataset = _pqt.ParquetFile(self.xpath)
        nbatches = pack_dataset.num_row_groups
        print(f"Dataset has {nbatches} batches, analyzing first batch...")
        bdf = pack_dataset.read_row_group(0).to_pandas()
        pack_dataset.close()

        packs = self.splitPdfsByPid(bdf)
        print(f"Found {len(packs)} packs in first batch, showing first one:")
        for pid, pk in packs:
            pk = self.finalizepdf(pk)
            mlb = self.mlbByPid(pid)
            print(f" - PackId: {pid}\n - NumRows: {len(pk)}\n - MLB: {mlb}\n - Sample Data:\n{pk.head(num)}")
            #print(f"PackId: {pid}, MLB: ??\nPACK:\n{pk.head(num)}\n")
            return
    
    def save(self):
        print(f"Processing and Saving packs as Graphs...")
        nprocs = _mp.cpu_count() // 2
        print(f"Building graphs using {nprocs} processes...")

        if self.gpath.exists():
            _rmrf(self.gpath)
        self.gpath.mkdir(parents=True, exist_ok=True)

        progress_queue = _Queue()
        data_src_queue = _Queue(maxsize=nprocs * 2)
        processes: list[_mp.Process] = []

        for i in range(nprocs):
            p = _mp.Process(target=pack2graph, kwargs={
                'vinfo_df': self.vinfo_df,
                'm_radius': self.m_radius,
                'active_labels': self.active_labels if hasattr(self, 'active_labels') else None,
                'gpath': self.gpath,
                'progress_queue': progress_queue,
                'data_src_queue': data_src_queue
            })
            p.start()
            processes.append(p)

        logproc = _mp.Process(target=logworker, kwargs={
            'progress_queue': progress_queue,
            'total': len(self.labels_df['PackId'].unique()) if hasattr(self, 'labels_df') else None
        })
        logproc.start()

        pack_dataset = _pqt.ParquetFile(self.xpath)
        nbatches = pack_dataset.num_row_groups
        for i in range(nbatches):
            bdf = pack_dataset.read_row_group(i).to_pandas()
            packs = self.splitPdfsByPid(bdf)

            while len(packs) > 0:
                pid0,pk0 = packs.pop(0)
                pk0 = self.finalizepdf(pk0)
                mlb0 = self.mlbByPid(pid0)
                data_src_queue.put((pid0, pk0, mlb0))

        pack_dataset.close()

        # signal the workers to stop
        for _ in range(nprocs):
            data_src_queue.put(None)

        for p in processes:
            p.join()
        logproc.join()
        print(f"All graphs built and saved to {self.gpath}")
                

