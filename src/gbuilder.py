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
import json as _json

from .labels import LabelsEnum as _LBEN

def rescaleToCenter(x_arr:_np.ndarray,dims_arr:_np.ndarray)->_np.ndarray:
    """
    Rescales the (X,Y) coordinates in the graph data based on the vehicle dimensions (width, length) and on the angle.
    Assuming original coordinates are taken from the center of the front border of the vehicle box,
    new coordinates will be in the center of the vehicle box.
    """
    x = x_arr.copy()
    xs = x[:,:,0]
    ys = x[:,:,1]
    angles = x[:,:,3]
    lengths = dims_arr[:,:,1]  # length is second static feature


    # apply offsets
    # TODO:CHECK - in all angle usage, check deg/rad consistency with input data
    x[:,:,0] = xs - (lengths / 2) * _np.cos(angles)
    x[:,:,1] = ys - (lengths / 2) * _np.sin(angles)

    return x

def pack2graph(frames_num:int,*,vinfo_df:_pd.DataFrame,m_radius:float,active_labels:list[int]=None,gpath:_Path,progress_queue:_Queue,data_src_queue:_Queue):

    if active_labels is None:
        active_labels = [le.value for le in _LBEN]
    
    vinfo_df['stType'] = vinfo_df['stType'].astype('long')

    t_fnames = ['X','Y','Speed','Angle','PresenceFlag']
    t_fnum = len(t_fnames)
    t_fnum_final = t_fnum + 1 # heading is encoded as sin+cos, so it adds one feature
    # final structure is [X,Y,Speed,(Angle)|(HeadingSin,HeadingCos),PresenceFlag]

    st_fnames = ['width','length','stType']
    st_fnum = len(st_fnames)

    stt_fidx = t_fnum + st_fnum -1  # index of stType in total features

    tot_fnames = t_fnames + st_fnames
    tot_fnum = t_fnum + st_fnum

    item = data_src_queue.get()
    while item is not None:
        gdata_dict = dict()
        pack_id, pack_df, mlb = item

        # add static features from vinfo
        pack_df = pack_df.merge(vinfo_df, on='VehicleId', how='inner')
        pack_df = pack_df.sort_values(['VehicleId','FrameId'])
        raw_feats = pack_df[tot_fnames].to_numpy().reshape(-1, frames_num, tot_fnum)  # num_vehicles x num_frames x num_features

        x = raw_feats[:,:,:t_fnum] # temporal features
        # set angles to rad
        x[:,:,3] = _np.deg2rad( x[:,:,3] )

        xdims = raw_feats[:,0:1,t_fnum:stt_fidx] # static features (same for all frames)
        xsttype = raw_feats[:,0,stt_fidx]

        x = rescaleToCenter(x, xdims)
        x  = _tch.tensor(x, dtype=_tch.float, device='cpu')
        xsttype = _tch.tensor(xsttype, dtype=_tch.long, device='cpu').flatten()

        #TODO:CHECK THIS
        # now we can reshape
        xdims = xdims.reshape(xdims.shape[0], -1)  # num_vehicles x num_static_features
        xdims = _tch.tensor(xdims, dtype=_tch.float, device='cpu')
        
        
        # edge index construction based on distance of trajectories
        ## min distance used for threshold
        ## inverse of avg distance used for edge values
        edge_index_list = []
        edge_attr_list = []
        num_vehicles = x.shape[0]
        for i in range(num_vehicles):
            xi = x[i,:,:2]  # X,Y - (fr, 2)
            pi = x[i,:,4]  # PresenceFlag - (fr,)
            for j in range(num_vehicles):
                if i != j:
                    xj = x[j,:,:2]  # X,Y - (fr, 2)
                    pj = x[j,:,4]  # PresenceFlag - (fr,)

                    # compute distances
                    dists:_np.ndarray = _np.linalg.norm(xi - xj, axis=1)  # (fr,)
                    # consider only frames where both vehicles are present
                    presence_mask = (pi > 0.5) & (pj > 0.5)
                    dists = dists[presence_mask]
                    if (not dists.size == 0) and dists.min() <= m_radius:
                        min_dist = dists.min()
                        max_dist = dists.max()
                        mean_dist = dists.mean()
                        msq_dist = (dists ** 2).mean()

                        edge_index_list.append([i,j])
                        # use all attributes for distance statistics
                        edge_attr_list.append([min_dist, max_dist, mean_dist, msq_dist])

        gdata_dict['edge_index'] = _tch.tensor(edge_index_list, dtype=_tch.long).t().contiguous() if len(edge_index_list) > 0 else _tch.empty((2,0), dtype=_tch.long)
        gdata_dict['edge_attr'] = _tch.tensor(edge_attr_list, dtype=_tch.float) if len(edge_attr_list) > 0 else _tch.empty((0,4), dtype=_tch.float)

        # -- HEADING ENCODED: replace angle with 2 features of sin+cos heading encoding
        h = x[:,:,3:4]
        hsin = _tch.sin(h)  # (num_vehicles, num_frames, 1)
        hcos = _tch.cos(h)  # (num_vehicles, num_frames, 1)
        x = _tch.cat([x[:,:,:3], hsin, hcos, x[:,:,4:]], dim=2)
        
        gdata_dict['x'] = x
        gdata_dict['xsttype'] = xsttype
        gdata_dict['xdims'] = xdims
        
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

class GraphOnlineCreator:
    def __init__(self,frames_num:int, m_radius:float, active_labels:list[int]|None, *,has_label:bool)->_GData:
        self.frames_num = frames_num
        self.m_radius = m_radius
        self.active_labels = active_labels if active_labels is not None else [le.value for le in _LBEN]
        self.has_label = has_label

        # ========================= field indexing =========================
        self.t_fnames = ['X','Y','Speed','Angle','PresenceFlag']
        self.t_fnum = len(self.t_fnames)
        self.t_fnum_final = self.t_fnum + 1 # as heading is encoded as sin+cos, it adds one feature to the original angle feature
        # final structure is [X,Y,Speed,(Angle)|(HeadingSin,HeadingCos),PresenceFlag]
        self.st_fnames = ['width','length','stType']
        self.st_fnum = len(self.st_fnames)
        self.stt_fidx = self.t_fnum + self.st_fnum -1  # index of stType in total features
        self.tot_fnames = self.t_fnames + self.st_fnames
        self.tot_fnum = self.t_fnum + self.st_fnum

    def finalizePdf(self, pack_df:_pd.DataFrame)->_pd.DataFrame:
        """ Zero-Padding also here """
        pack_df['PresenceFlag'] = 1.0
        pack_df['PresenceFlag'] = pack_df['PresenceFlag'].astype('float16')

        for vid, vg in pack_df.groupby('VehicleId'):
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
                        'PresenceFlag': 0.0,
                        'width': vg['width'].iloc[0],
                        'length': vg['length'].iloc[0],
                        'stType': vg['stType'].iloc[0]
                    }])
                    new_row = new_row.astype(pack_df.dtypes.to_dict())
                    pack_df = _pd.concat([pack_df, new_row], ignore_index=True)

        # sort by FrameId, VehicleId for consistent ordering
        pack_df = pack_df.sort_values(['FrameId', 'VehicleId']).reset_index(drop=True)
        return pack_df


    def __call__(self, full_pack_df:_pd.DataFrame, mlb:int|None=None)->_GData:
        if self.has_label and mlb is None:
            raise ValueError("GraphOnlineCreator is configured to expect labels, but mlb argument is None")

        # ========================= nodes =========================

        gdata_dict = dict()
        pack_df = self.finalizePdf(full_pack_df).copy().sort_values(['VehicleId','FrameId'])
        raw_feats = pack_df[self.tot_fnames].to_numpy().reshape(-1, self.frames_num, self.tot_fnum)  # num_vehicles x num_frames x num_features

        x = raw_feats[:,:,:self.t_fnum] # temporal features
        # set angles to rad
        x[:,:,3] = _np.deg2rad( x[:,:,3] )

        xdims = raw_feats[:,0:1,self.t_fnum:self.stt_fidx] # static features (same for all frames)
        xsttype = raw_feats[:,0,self.stt_fidx]

        x = rescaleToCenter(x, xdims)
        x  = _tch.tensor(x, dtype=_tch.float, device='cpu')
        xsttype = _tch.tensor(xsttype, dtype=_tch.long, device='cpu').flatten()

        # remove width and length (first 2 columns) from static features
        xdims = xdims.reshape(xdims.shape[0], -1)  # num_vehicles x num_static_features
        xdims = _tch.tensor(xdims, dtype=_tch.float, device='cpu')
        
        # ========================= edge construction =========================

        # edge index construction based on distance of trajectories
        ## min distance used for threshold
        ## inverse of avg distance used for edge values
        edge_index_list = []
        edge_attr_list = []
        num_vehicles = x.shape[0]
        for i in range(num_vehicles):
            xi = x[i,:,:2]  # X,Y - (fr, 2)
            pi = x[i,:,4]  # PresenceFlag - (fr,)
            for j in range(num_vehicles):
                if i != j:
                    xj = x[j,:,:2]  # X,Y - (fr, 2)
                    pj = x[j,:,4]  # PresenceFlag - (fr,)

                    # compute distances
                    dists:_np.ndarray = _np.linalg.norm(xi - xj, axis=1)  # (fr,)
                    # consider only frames where both vehicles are present
                    presence_mask = (pi > 0.5) & (pj > 0.5)
                    dists = dists[presence_mask]
                    if (not dists.size == 0) and dists.min() <= self.m_radius:
                        min_dist = dists.min()
                        max_dist = dists.max()
                        mean_dist = dists.mean()
                        msq_dist = (dists ** 2).mean()

                        edge_index_list.append([i,j])
                        # use all attributes for distance statistics
                        edge_attr_list.append([min_dist, max_dist, mean_dist, msq_dist])

        gdata_dict['edge_index'] = _tch.tensor(edge_index_list, dtype=_tch.long).t().contiguous() if len(edge_index_list) > 0 else _tch.empty((2,0), dtype=_tch.long)
        gdata_dict['edge_attr'] = _tch.tensor(edge_attr_list, dtype=_tch.float) if len(edge_attr_list) > 0 else _tch.empty((0,4), dtype=_tch.float)


        # ========================= finalize features =========================

        # replace angle with 2 features of sin+cos heading encoding
        h = x[:,:,3:4]
        hsin = _tch.sin(h)  # (num_vehicles, num_frames, 1)
        hcos = _tch.cos(h)  # (num_vehicles, num_frames, 1)
        x = _tch.cat([x[:,:,:3], hsin, hcos, x[:,:,4:]], dim=2)

        gdata_dict['x'] = x
        gdata_dict['xsttype'] = xsttype
        gdata_dict['xdims'] = xdims
        
        if mlb is not None:
            # labels are stored as bitmask in an integer
            y = _tch.zeros((len(self.active_labels),), dtype=_tch.float)
            for i,c in enumerate(self.active_labels):
                if mlb & (1 << c):
                    y[i] = 1.0 
            # return graph data object
            gdata_dict['y'] = y

        gdata = _GData(**gdata_dict)
        return gdata        
        
class MapBuilder:
    filepath: _Path
    def __init__(self, filepath:_Path, lat_conn_max_angle_deg:float, lat_conn_proximity_threshold:float):
        self.filepath = filepath.resolve()
        self.lat_conn_max_angle_deg = lat_conn_max_angle_deg
        self.lat_conn_proximity_threshold = lat_conn_proximity_threshold
        savedir = self.filepath.parent / '.map'
        if not savedir.exists():
            savedir.mkdir(parents=True, exist_ok=True)
        self.savepath = savedir / (self.filepath.stem + '.pth')

    @staticmethod
    def segmentsAngles(start_tensor:_tch.Tensor,end_tensor:_tch.Tensor)->_tch.Tensor:
        """ Computes the angle of each segment based on start and end coordinates. Shapes are [NUM_SEGMENTS, 2] for both inputs """
        delta_x = end_tensor[:,0] - start_tensor[:,0]
        delta_y = end_tensor[:,1] - start_tensor[:,1]
        angles = _np.arctan2(delta_y, delta_x)  # returns angle in radians
        return angles
    
    @staticmethod
    def segmentsDistance(seg1_start:_tch.Tensor,seg1_end:_tch.Tensor, width1:_tch.Tensor, seg2_start:_tch.Tensor,seg2_end:_tch.Tensor, width2:_tch.Tensor)->_tch.Tensor:
        l1 = _np.linalg.norm(seg1_end - seg1_start)
        l2 = _np.linalg.norm(seg2_end - seg2_start)

        # compute with l1 as longest one, so if l2>l1 swap
        if l2 > l1:
            return MapBuilder.segmentsDistance(seg2_start,seg2_end,width2, seg1_start,seg1_end,width1)
        else:
            #TODO: CHECK IF THIS IS THE BEST WAY TO COMPUTE DISTANCE BETWEEN SEGMENTS
            # raw distance is computed as orth distance from centroid2 to segment1
            # segment1: A-B
            # centroid2: C
            C = (seg2_start + seg2_end) / 2.0
            A = seg1_start
            B = seg1_end
            AC = C-A
            AB = B-A
            #BC = C-B

            # compute projection of C in AB: P=A + t*AB,t=AC@AB/AB@AB (but clamped to segment extents)
            t = _np.dot(AC, AB) / _np.dot(AB, AB)
            if t < 0.0:
                P = A
            elif t > 1.0:
                P = B
            else:
                P = A + t * AB

            # compute raw distance as distance Centroid-Projection
            raw_dist = _np.linalg.norm(C - P)

            # adjusted distance considering widths
            return raw_dist - (width1 + width2) / 2.0


    
    def save(self):
        df = _pd.read_parquet(self.filepath).astype({
            'start_x':'float32',
            'start_y':'float32',
            'end_x':'float32',
            'end_y':'float32',
            'lane_type':'uint8',
            'speed_limit':'float32',
            'width':'float32',
            'can_go_left':'bool',
            'can_go_right':'bool'
        })

        #print(f"df:{df.head(20)}")

        # convert to torch tensor
        #TODO ADD EXPLICIT ORDER OF COLUMNS
        float_features = _tch.tensor(df.drop(columns=['lane_type', 'can_go_left', 'can_go_right']).to_numpy(dtype=_np.float32), dtype=_tch.float)
        bool_features = _tch.tensor(df[['can_go_left','can_go_right']].to_numpy(dtype=_np.bool_), dtype=_tch.bool)
        lane_type_cats = _tch.tensor(df['lane_type'].to_numpy(dtype=_np.long), dtype=_tch.long)

        # compute centroids
        start_coords = float_features[:, 0:2]  # start_x, start_y
        end_coords = float_features[:, 2:4]    # end_x, end_y
        cgl = bool_features[:, 0:1]  # can_go_left
        cgr = bool_features[:, 1:2]  # can_go_right
        widths = float_features[:, 5:6]  # width
        angles = self.segmentsAngles(start_coords,end_coords).unsqueeze(1)  # [NUM_SEGMENTS, 1]
        centroids = (start_coords + end_coords) / 2.0  # [NUM_SEGMENTS, 2]

        # build map graph (edge indexes)
        edge_indexes = []
        num_segments = float_features.shape[0]
        for i in _tqdm(range(num_segments), desc="Building Map Graph Edges...", unit="segments"):
            start_i = start_coords[i]
            end_i = end_coords[i]
            ang_i = angles[i]
            cgl_i = cgl[i]
            cgr_i = cgr[i]
            w_i = widths[i]
            for j in range(num_segments):
                if i != j:
                    start_j = start_coords[j]
                    end_j = end_coords[j]
                    ang_j = angles[j]
                    cgl_j = cgl[j]
                    cgr_j = cgr[j]
                    w_j = widths[j]
                    fwd_dist = _np.linalg.norm(end_i.numpy() - start_j.numpy())

                    # LANE-LATERAL NEIGH connections: check angle similarity and proximity
                    delta_angle = _np.abs(ang_i - ang_j).item()
                    if delta_angle < _np.deg2rad(self.lat_conn_max_angle_deg):  # approximately same direction
                        if self.segmentsDistance(start_i,end_i,w_i,start_j,end_j,w_j).item() < self.lat_conn_proximity_threshold:
                            #FIXME: use connections only if lanes are directionally compatible
                            # L-R connections
                            if cgr_i.item() and cgl_j.item():
                                edge_indexes.append([i, j])
                            # R-L connections
                            elif cgl_i.item() and cgr_j.item():
                                edge_indexes.append([i, j])

                    # END-START FWD connections
                    # check if segment i's end is close to segment j's start
                    elif fwd_dist < 2.0:  # threshold distance to consider a connection
                        edge_indexes.append([i, j])
        edge_indexes = _tch.tensor(edge_indexes, dtype=_tch.long).t().contiguous()  # [2, NUM_EDGES]


        if self.savepath.exists():
            self.savepath.unlink()
        _tch.save({'float_features': float_features, 'bool_features': bool_features, 'lane_type_cats': lane_type_cats, 'mseg_centroids': centroids, 'mgraph_edge_indexes': edge_indexes}, self.savepath)

class GraphsBuilder:
    dirpath: _Path # path to the dataset directory
    gpath: _Path  # path to output graphs directory
    frames_num: int # number of frames per pack
    m_radius: float # threshold radius for edge connection
    xpath: _Path # path to packs.parquet
    ypath: _Path # path to labels.parquet
    vpath: _Path # path to vinfo.parquet
    active_labels: list[int] # list of active label indices
    labels_df: _pd.DataFrame # dataframe of labels
    vinfo_df: _pd.DataFrame # dataframe of vehicle info

    def __init__(self,dirpath:_Path,*,frames_num:int,m_radius:float, active_labels:list[int]=None):

        self.dirpath = dirpath.resolve()
        self.gpath = self.dirpath / '.graphs' # output graphs path

        self.frames_num = frames_num
        self.m_radius = m_radius

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
        # TODO HERE WE CAN ALSO SET SINGLE CHECK AT INIT FOR LABELS.PQT PRESENCE OR NOT
        if hasattr(self, 'labels_df'):
            row = self.labels_df[self.labels_df['PackId'] == pid]
            if len(row) > 1:
                raise ValueError(f"Multiple label rows found for PackId {pid}")
            elif len(row) == 1:
                return int(row['MLBEncoded'].values[0])
        return None
    
    def save(self):
        """
        Processes the packs and saves them as graph data files.
        Graphs are saved in the directory specified by self.gpath.
        Format of saved data is:
        - `.x`: Node feature matrix, of shape [num_vehicles, num_frames, *NTF*].
        - `.xsttype`: Static feature vector of vehicle types, of shape [num_vehicles]
        - `.edge_index`: Edge index tensor, of shape [2, num_edges] (if aggregateEdges is True)
        - `.edge_attr`: Edge attribute tensor, of shape [num_edges, 1] (if aggregateEdges is True)
        - `.edge_index_all`: Frame-based concatenation (dim=1) of edge index tensors, one per frame (if aggregateEdges is False)
        - `.edge_attr_all`: Frame-based concatenation (dim=0) of edge attribute tensors, one per frame (if aggregateEdges is False)
        - `.edge_frame_ptrs`: Tensor of edge frame pointers of shape [NFrames+1] (if aggregateEdges is False or flattenTimeAsGraphs is True)
        - `.node_frame_ptrs`: Tensor of node frame pointers of shape [NFrames+1] (if flattenTimeAsGraphs is True)
        - `.y`: Multi-label binary vector of shape [num_active_labels] (if labels are present)

        Note that featues used in nodes are:
        - Temporal features per vehicle per frame (*NTF*): [X, Y, Speed, (HeadingSin, HeadingCos) | Angle, PresenceFlag]
        """
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

        n_samples = self.labels_df['PackId'].nunique() if (hasattr(self, 'labels_df') and self.labels_df is not None) else len(list(self.gpath.glob('*.pt')))
        n_positive = None
        if hasattr(self, 'labels_df') and self.labels_df is not None:
            n_positive = int((self.labels_df['MLBEncoded'] > 0).sum())
        
        # save metadata
        meta_dict = {
            'n_samples': n_samples,
            'n_positive': n_positive,
            'n_edge_features': 4,
            'frames_num': self.frames_num,
            'm_radius': self.m_radius,
            'active_labels': self.active_labels
        }
        with open(self.gpath / 'metadata.json', 'w', encoding='utf-8') as metafile:
            _json.dump(meta_dict, metafile, indent=4, ensure_ascii=False)

                

