import multiprocessing as _mp
from multiprocessing import Queue as _Queue
import pandas as _pd
import numpy as _np
import torch as _tch
from torch_geometric.data import Data as _GData
from pathlib import Path as _Path
from shutil import rmtree as _rmrf
from tqdm.auto import tqdm as _tqdm
import pyarrow.parquet as _pqt

from .labels import LabelsEnum as _LBEN

def rescaleToCenter(x_arr:_np.ndarray,stat_arr:_np.ndarray)->_np.ndarray:
    """
    Rescales the (X,Y) coordinates in the graph data based on the vehicle dimensions (width, length) and on the angle.
    Assuming original coordinates are taken from the center of the front border of the vehicle box,
    new coordinates will be in the center of the vehicle box.
    """
    x = x_arr.copy()
    xs = x[:,:,0]
    ys = x[:,:,1]
    angles = x[:,:,3]
    lengths = stat_arr[:,:,1]  # length is second static feature


    # apply offsets
    x[:,:,0] = xs - (lengths / 2) * _np.cos(angles)
    x[:,:,1] = ys - (lengths / 2) * _np.sin(angles)

    return x

def pack2graph(frames_num:int,*,vinfo_df:_pd.DataFrame,m_radius:float,active_labels:list[int]=None,gpath:_Path,progress_queue:_Queue,data_src_queue:_Queue, addSinCosTimeEnc:bool=False, rscToCenter:bool=True, removeDims:bool=False, flattenTime:bool=False)->_GData:

    if active_labels is None:
        active_labels = [le.value for le in _LBEN]
    
    vinfo_df['stType'] = vinfo_df['stType'].astype('long')

    t_dims = ['X','Y','Speed','Angle','PresenceFlag']
    st_dims = ['width','length','stType']

    if addSinCosTimeEnc:
            # already added in finalizepdf
            t_dims.extend(['tsin', 'tcos'])
    tot_fnames = t_dims + st_dims
    
    temp_dnum = len(t_dims)
    stat_dnum = len(st_dims)
    tot_dnum = temp_dnum + stat_dnum

    item = data_src_queue.get()
    while item is not None:
        pack_id, pack_df, mlb = item

        # add static features from vinfo
        pack_df = pack_df.merge(vinfo_df, on='VehicleId', how='inner')
        pack_df = pack_df.sort_values(['VehicleId','FrameId'])
        raw_feats = pack_df[tot_fnames].to_numpy().reshape(-1, frames_num, tot_dnum)  # num_vehicles x num_frames x num_features

        x = raw_feats[:,:,:temp_dnum] # temporal features
        statx = raw_feats[:,0:1,temp_dnum:] # static features (same for all frames)

        if rscToCenter:
            x = rescaleToCenter(x, statx)

        if flattenTime:
            x = x.reshape(-1, temp_dnum*frames_num)  # flatten over time dimension

        if removeDims:
            # remove width and length from static features
            statx = _np.delete(statx, [0,1], axis=2 if flattenTime else 1)  # remove first two columns (width, length)

        gdata_dict = {
            'x': _tch.tensor(x, dtype=_tch.float, device='cpu'),
            'statx': _tch.tensor(statx, dtype=_tch.float, device='cpu')
        }
        
        
        if flattenTime:
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
                        if dists.mean() <= m_radius:
                            avg_dist = _np.mean(dists[_np.isfinite(dists)])
                            edge_index_list.append([i,j])
                            edge_attr_list.append([1.0 / (avg_dist + 1e-6)])  # avoid div by zero

            gdata_dict['edge_index'] = _tch.tensor(edge_index_list, dtype=_tch.long).t().contiguous()
            gdata_dict['edge_attr'] = _tch.tensor(edge_attr_list, dtype=_tch.float)
        else:
            # all parameters are per-frame, so edge_index and edge_attr are computed per-frame and concatenated in lists
            edge_index_list = []
            edge_attr_list = []
            num_vehicles = x.shape[0]
            for f in range(frames_num):
                eil_internal_frame = []
                eal_internal_frame = []
                for i in range(num_vehicles):
                    if x[i,f,4] < 0.5:
                        continue # skip if not present
                    xi = x[i,f,0:2]  # X,Y
                    for j in range(num_vehicles):
                        if x[j,f,4] < 0.5:
                            continue # skip if not present
                        if i != j:
                            xj = x[j,f,0:2]
                            dist = _np.linalg.norm(xi - xj)
                            if dist <= m_radius:
                                eil_internal_frame.append([i,j])
                                eal_internal_frame.append([1.0 / (dist + 1e-6)])  # avoid div by zero
                edge_index_list.append(_tch.tensor(eil_internal_frame, dtype=_tch.long).t().contiguous())
                edge_attr_list.append(_tch.tensor(eal_internal_frame, dtype=_tch.float))
            gdata_dict['edge_index_list'] = edge_index_list
            gdata_dict['edge_attr_list'] = edge_attr_list

        if mlb is not None:
            # labels are stored as bitmask in an integer
            y = _tch.zeros((len(active_labels),), dtype=_tch.float)
            for i,c in enumerate(active_labels):
                if mlb & (1 << c):
                    y[i] = 1.0 
            # return graph data object
            gdata_dict['y'] = y

        gdata = _GData(**gdata_dict)
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
    dirpath: _Path # path to the dataset directory
    gpath: _Path  # path to output graphs directory
    frames_num: int # number of frames per pack
    m_radius: float # threshold radius for edge connection
    addSinCosTimeEnc: bool # whether to add sin/cos time encoding features
    rscToCenter: bool # whether to rescale coordinates to vehicle center
    removeDims: bool # whether to remove vehicle dimension features
    xpath: _Path # path to packs.parquet
    ypath: _Path # path to labels.parquet
    vpath: _Path # path to vinfo.parquet
    active_labels: list[int] # list of active label indices
    labels_df: _pd.DataFrame # dataframe of labels
    vinfo_df: _pd.DataFrame # dataframe of vehicle info

    def __init__(self,dirpath:_Path,*,frames_num:int,m_radius:float, addSinCosTimeEnc:bool=True, rscToCenter:bool=False, removeDims:bool=False, active_labels:list[int]=None):

        self.dirpath = dirpath.resolve()
        self.gpath = self.dirpath / '.graphs' # output graphs path

        self.frames_num = frames_num
        self.m_radius = m_radius

        self.addSinCosTimeEnc = addSinCosTimeEnc
        self.rscToCenter = rscToCenter
        self.removeDims = removeDims

        self.xpath = self.dirpath / 'packs.parquet'
        self.ypath = self.dirpath / 'labels.parquet'
        self.vpath = self.dirpath / 'vinfo.parquet'

        if active_labels is None:
            # all
            active_labels = [le.value for le in _LBEN]
        elif len(active_labels) == 0:
            raise ValueError("active_labels must contain at least one label index")
        else:
            for c in active_labels:
                if not isinstance(c, int) or c < 0:
                    raise ValueError("active_labels must contain only non-negative integers")
            self.active_labels = active_labels


        # load y df if available
        if self.ypath.exists() and self.ypath.is_file():
            self.labels_df = _pd.read_parquet(self.ypath).astype({'PackId':'uint32','MLBEncoded':'uint16'})

        # load vinfo df if available
        if self.vpath.exists() and self.vpath.is_file():
            self.vinfo_df = _pd.read_parquet(self.vpath).astype({'VehicleId':'string', 'width':'float32', 'length':'float32', 'stType':'uint8'})
            # set NaN w/l to 0.0
            self.vinfo_df['width'] = self.vinfo_df['width'].fillna(0.0)
            self.vinfo_df['length'] = self.vinfo_df['length'].fillna(0.0)

    def finalizepdf(self,packdf:_pd.DataFrame)->_pd.DataFrame:
        """ Pandas preprocessing that finalizes the pack DataFrame by adding missing frames, PresenceFlag, and time encoding features in order to be used for as structured tensors. """
        
        pdf = packdf.copy()

        # add PresenceFlag
        pdf['PresenceFlag'] = 1.0
        pdf['PresenceFlag'] = pdf['PresenceFlag'].astype('float16')

        # drop PackId column as it's no longer needed for tensors (external info only, available in filename)
        pdf.drop(columns=['PackId'], inplace=True)

        # add missing frames with zeroed features and PresenceFlag=0.0
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
        
        if self.addSinCosTimeEnc:
            # apply sin+cos temporal encoding as 2 extra features
            pdf['tsin'] = pdf['FrameId'].map(lambda x: _np.sin(2 * _np.pi * x / self.frames_num)).astype('float16')
            pdf['tcos'] = pdf['FrameId'].map(lambda x: _np.cos(2 * _np.pi * x / self.frames_num)).astype('float16')
        
        # sort by FrameId, VehicleId for consistent ordering
        pdf = pdf.sort_values(['FrameId', 'VehicleId']).reset_index(drop=True)

        return pdf

    @staticmethod
    def getPDFList(concat_pdfs:_pd.DataFrame)->list[tuple[int,_pd.DataFrame]]:
        """
        Takes a batch df and splits it into a list of Pack DataFrames, one per PackId.
        """
        packs = []
        for pid, pg in concat_pdfs.groupby('PackId'):
            packs.append((pid, pg))
        return packs
    
    def mlbByPid(self,pid:int)->int|None:
        """ Returns the MLBEncoded label for the given PackId, or None if not found. """
        # HERE WE CAN ALSO SET SINGLE CHECK AT INIT FOR LABELS.PQT PRESENCE OR NOT
        if hasattr(self, 'labels_df'):
            row = self.labels_df[self.labels_df['PackId'] == pid]
            if len(row) > 1:
                raise ValueError(f"Multiple label rows found for PackId {pid}")
            elif len(row) == 1:
                return int(row['MLBEncoded'].values[0])
        return None
    
    def save(self):
        """ Processes the packs and saves them as graph data files. """
        nprocs = _mp.cpu_count() // 2
        print(f"Processing and Saving packs as Graphs, using {nprocs} processes...")

        if self.gpath.exists():
            _rmrf(self.gpath)
        self.gpath.mkdir(parents=True, exist_ok=True)

        progress_queue = _Queue()
        data_src_queue = _Queue(maxsize=nprocs * 2)
        processes: list[_mp.Process] = []

        for i in range(nprocs):
            p = _mp.Process(target=pack2graph, kwargs={
                'frames_num': self.frames_num,
                'vinfo_df': self.vinfo_df,
                'm_radius': self.m_radius,
                'active_labels': self.active_labels if hasattr(self, 'active_labels') else None,
                'gpath': self.gpath,
                'progress_queue': progress_queue,
                'data_src_queue': data_src_queue,
                'rscToCenter': self.rscToCenter,
                'removeDims': self.removeDims,
                'addSinCosTimeEnc': self.addSinCosTimeEnc
            })
            p.start()
            processes.append(p)

        logproc = _mp.Process(target=logworker, kwargs={
            'progress_queue': progress_queue,
            'total': len(self.labels_df['PackId'].unique()) if hasattr(self, 'labels_df') else None
        })
        logproc.start()

        # dispatch X,Y data to workers
        pack_dataset = _pqt.ParquetFile(self.xpath)
        nbatches = pack_dataset.num_row_groups
        for i in range(nbatches):
            batch_df = pack_dataset.read_row_group(i).to_pandas()
            packs = self.getPDFList(batch_df)

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
                

