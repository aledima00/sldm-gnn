import torch as _torch
import torch.nn as _nn
from torch_geometric.nn import global_mean_pool as _gmean_pool, global_max_pool as _gmax_pool
from typing import Literal as _Lit

from .blocks.sageblock import SageBlock as _SageBlock

class FrameSubgraphPoolingFunction:
    def __init__(self, batch_size:int,num_frames:int, global_pooling:_Lit['mean', 'max','double']='double'):
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.double = False
        match global_pooling:
            case 'mean':
                self.pooling_fn = _gmean_pool
            case 'max':
                self.pooling_fn = _gmax_pool
            case 'double':
                self.pooling_fn = lambda x,batch: _torch.cat([_gmean_pool(x, batch), _gmax_pool(x, batch)], dim=1)
                self.double=True
            case _:
                raise ValueError(f"Unsupported global_pooling method: {global_pooling}")
            
    def __call__(self,x:_torch.Tensor, node_frame_ptrs:_torch.Tensor, batchptr:_torch.Tensor):
        """ Pool node features x per frame using edge_frame_ptrs to identify subgraphs corresponding to different frames.
            x: Tensor of shape (num_nodes, num_features)
            node_frame_ptrs: List of indices indicating the start of each frame in the node index (size [num_frames + 1,])
            batchptr: Tensor indicating batch pointers
        """
        pooled_batches = []
        for i in range(batchptr.shape[0]-1):
            # batch node indices
            b_start_idx = batchptr[i].item()
            b_end_idx = batchptr[i+1].item()

            # batch-specific data
            b_x_raw = x[b_start_idx:b_end_idx, :]

            # frame-batch indexes and data - (nframes+1) * batch_num
            nfp_start = (self.num_frames+1)*i
            nfp_end = (self.num_frames+1)*(i+1)
            b_node_frame_ptrs = node_frame_ptrs[nfp_start:nfp_end]
            # already relative to batch start, so no need to adjust

            braw_len = b_x_raw.shape[0]


            # prappare results for this batch
            frames_fake_batch_raw = _torch.zeros((braw_len,), dtype=_torch.long, device=x.device)
            insert_list = []

            for j in range(self.num_frames):
                start_idx = b_node_frame_ptrs[j] # node index where frame j starts
                end_idx = b_node_frame_ptrs[j + 1] # node index where frame j ends
                if start_idx == end_idx:
                    # add zeroed one-node graph to avoid empty frame
                    insert_list.append(j)
                else:
                    frames_fake_batch_raw[start_idx:end_idx] = j
                
            
            # insert empty frames
            if len(insert_list) > 0:
                b_x = _torch.zeros((braw_len + len(insert_list), b_x_raw.shape[1]), device=b_x_raw.device, dtype=b_x_raw.dtype)
                frames_fake_batch = _torch.zeros((braw_len + len(insert_list),), dtype=_torch.long, device=x.device)
                cnt = 0
                for i in range(self.num_frames):
                    start_idx_src = b_node_frame_ptrs[i]
                    end_idx_src = b_node_frame_ptrs[i + 1]
                    start_idx = start_idx_src + cnt
                    end_idx = end_idx_src + cnt
                    if i in insert_list:
                        # insert zeroed one-node graph
                        b_x[start_idx:start_idx+1, :] = 0.0
                        frames_fake_batch[start_idx:start_idx+1] = i
                        cnt += 1
                    else:
                        b_x[start_idx:end_idx, :] = b_x_raw[start_idx_src:end_idx_src, :]
                        frames_fake_batch[start_idx:end_idx] = frames_fake_batch_raw[start_idx_src:end_idx_src]
            else:
                b_x = b_x_raw
                frames_fake_batch = frames_fake_batch_raw
                    


            #print(f"shape of b_xcat: {b_x.shape}, frames_fake_batch: {frames_fake_batch.shape}")

            # pool all frames for this batch

            # this trick allow to pool frames independently using fake batch indices
            pooled_frames = self.pooling_fn(b_x, frames_fake_batch)
            pooled_batches.append(pooled_frames)

        # # now concatenate all batches
        # for i in range(len(pooled_batches)):
        #     # pooled_batches[i] shape: (num_frames, num_features)
        #     print(f"Pooled batch {i} shape: {pooled_batches[i].shape}")
        return _torch.stack(pooled_batches, dim=0)  # shape: (batch_size, num_frames, num_features)
    

    def call_old(self,x:_torch.Tensor, node_frame_ptrs:_torch.Tensor, batchptr:_torch.Tensor):
        """ Pool node features x per frame using edge_frame_ptrs to identify subgraphs corresponding to different frames.
            x: Tensor of shape (num_nodes, num_features)
            node_frame_ptrs: List of indices indicating the start of each frame in the node index (size [num_frames + 1,])
            batchptr: Tensor indicating batch pointers
        """
        pooled_batches = []
        for i in range(batchptr.shape[0]-1):
            # batch node indices
            b_start_idx = batchptr[i].item()
            b_end_idx = batchptr[i+1].item()

            # batch-specific data
            b_x = x[b_start_idx:b_end_idx, :]

            # frame-batch indexes and data - (nframes+1) * batch_num
            nfp_start = (self.num_frames+1)*i
            nfp_end = (self.num_frames+1)*(i+1)
            b_node_frame_ptrs = node_frame_ptrs[nfp_start:nfp_end]
            # already relative to batch start, so no need to adjust

            # prappare results for this batch
            frames_fake_batch_list = []
            bxlist = []

            for j in range(self.num_frames):
                start_idx = b_node_frame_ptrs[j] # node index where frame j starts
                end_idx = b_node_frame_ptrs[j + 1] # node index where frame j ends
                if start_idx == end_idx:
                    # add zeroed one-node graph to avoid empty frame
                    b_xf = _torch.zeros((1, b_x.shape[1]), device=x.device)
                    f = _torch.full((1,), j, dtype=_torch.long, device=x.device)
                else:
                    b_xf = b_x[start_idx:end_idx, :]
                    f = _torch.full((end_idx - start_idx,), j, dtype=_torch.long, device=x.device)
                bxlist.append(b_xf)
                frames_fake_batch_list.append(f)
                
            
            # cat
            b_xcat = _torch.cat(bxlist, dim=0)
            frames_fake_batch = _torch.cat(frames_fake_batch_list, dim=0)

            # pool all frames for this batch

            # this trick allow to pool frames independently using fake batch indices
            pooled_frames = self.pooling_fn(b_xcat, frames_fake_batch)
            pooled_batches.append(pooled_frames)

        # # now concatenate all batches
        # for i in range(len(pooled_batches)):
        #     # pooled_batches[i] shape: (num_frames, num_features)
        #     print(f"Pooled batch {i} shape: {pooled_batches[i].shape}")
        return _torch.stack(pooled_batches, dim=0)  # shape: (batch_size, num_frames, num_features)
    




class SageGru(_nn.Module):
    def __init__(self, batch_size:int, dynamic_features_num:int, has_dims:bool, frames_num:int, sage_hidden_dims:list[int], fc1dims:list[int], gru_hidden_size:int, gru_num_layers:int, fc2dims:list[int]=[50,50], out_dim:int=1, num_st_types:int=256, emb_dim:int=12, dropout:float|None=None, negative_slope:float|None=None, global_pooling:_Lit['mean', 'max','double']='double'):
        super().__init__()

        #TODO validate inputs
        assert len(sage_hidden_dims) >= 1, "sage_hidden_dims must contain at least one element"
        # ...

        self.frames_num = frames_num

        # 1 - embedding for station types
        self.st_emb = _nn.Embedding(num_st_types, emb_dim)        

        # 2 - concat all input features
        last_step_dims = dynamic_features_num + (2 if has_dims else 0) + emb_dim
        self.has_dims = has_dims
        
        # 3 - GraphSAGE layers (per-frame)
        sage_dims = [last_step_dims] + sage_hidden_dims
        self.frame_sage = _SageBlock(sage_dims, dropout=dropout, negative_slope=negative_slope)
        last_step_dims = sage_dims[-1]

        # 4 - time resolution pooling
        self.timeres_pooling = FrameSubgraphPoolingFunction(batch_size=batch_size,num_frames=frames_num, global_pooling=global_pooling)
        if global_pooling == 'double':
            last_step_dims *= 2

        # 5 - fully connected layers before GRU
        ldims1 = [last_step_dims] + fc1dims
        self.fc1s = _nn.ModuleList([
            _nn.Sequential(
                _nn.Linear(ldims1[i], ldims1[i+1]),
                _nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else _nn.ReLU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(ldims1)-1)
        ])
        last_step_dims = ldims1[-1]

        # 6 - GRU layer to process dynamic features
        self.gru = _nn.GRU(
            input_size=last_step_dims,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True
        )
        last_step_dims = gru_hidden_size

        # 7 - fully connected layers after GRU
        ldims2 = [last_step_dims] + fc2dims
        self.fc2s = _nn.ModuleList([
            *[_nn.Sequential(
                _nn.Linear(ldims2[i], ldims2[i+1]),
                _nn.LeakyReLU(negative_slope=negative_slope) if negative_slope is not None else _nn.ReLU(),
                _nn.Dropout(p=dropout) if dropout is not None else _nn.Identity()
            ) for i in range(len(ldims2)-1)]
        ])
        last_step_dims = ldims2[-1]

        # 8 - final output layer
        self.linout = _nn.Linear(last_step_dims, out_dim)

    def forward(self, data):
        x, edge_index_all, edge_attr_all, node_frame_ptrs, xsttype, ptr = data.x, data.edge_index_all, data.edge_attr_all, data.node_frame_ptrs, data.xsttype, data.ptr

        # 1 - embedding for station types
        st_embedded:_torch.Tensor = self.st_emb(xsttype)

        # 2 - concat all input features
        if self.has_dims:
            xdims = data.xdims
            x = _torch.cat([x, xdims, st_embedded], dim=1)
        else:
            x = _torch.cat([x, st_embedded], dim=1)

        # 3 - process with GraphSAGE per frame
        x = self.frame_sage(x, edge_index_all)

        # 4 - time resolution pooling
        x = self.timeres_pooling(x, node_frame_ptrs, ptr)
        # Shape: (batch_size, frames_num, num_features)

        # 5 - process with fc layers before GRU
        for fc in self.fc1s:
            x = fc(x)

        # 6 - process dynamic features with GRU
        gru_out, hlast = self.gru(x)
        x = hlast[-1,:,:] # take last hidden state

        # 7 - fc layers after GRU
        for fc in self.fc2s:
            x = fc(x)

        # 8 - final output layer
        x = self.linout(x)
        return x







