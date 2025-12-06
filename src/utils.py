import torch as _tch
from torch.utils.data import random_split as _rsplit
from torch_geometric.loader import DataLoader as _GDL
from torch_geometric.data import Dataset as _GDataset
from typing import Literal as _Lit
from colorama import Fore as _Fore, Back as _Back, Style as _Style
from tqdm.auto import tqdm as _tqdm
import numpy as _np
from dataclasses import dataclass as _dc
from pathlib import Path as _Path
import json as _json

from .tprint import TabPrint as _TabPrint
from .labels import LabelsEnum as _LE

Progress_logging_options = _Lit['clilog', 'tqdm', 'none']

FmaskType = _Lit['x','y','pos','speed','heading','hsin','hcos']

@_dc
class MetaData:
    n_samples:int
    n_positive:int
    n_node_temporal_features:int
    n_edge_features:int
    frames_num:int
    m_radius:float
    sin_cos_time_enc:bool
    vpos_rescaled_center:bool
    has_dims:bool
    heading_encoded:bool
    aggregate_edges:bool
    active_labels:list[int]

    def getNegOverPosRatio(self) ->float:
        if self.n_positive == 0:
            raise ValueError("Number of positive samples is zero, cannot compute negative over positive ratio")
        n_negative = self.n_samples - self.n_positive
        return n_negative / self.n_positive

    @staticmethod
    def loadJson(path:_Path)->'MetaData':
        mddict:dict = None
        with open(path.resolve(), 'r', encoding='utf-8') as metafile:
            mddict = _json.load(metafile)
        return MetaData(**mddict)
    
    def getDataMask(self,selector:FmaskType)->_tch.Tensor:
        nfeats = 4 + (2 if self.heading_encoded else 1) + (2 if self.sin_cos_time_enc else 0)
        msk = _tch.full((nfeats,),False,dtype=_tch.bool)
        match selector:
            case 'x':
                msk[0] = True
            case 'y':
                msk[1] = True
            case 'pos':
                msk[0] = True
                msk[1] = True
            case 'speed':
                msk[2] = True
            case 'heading':
                msk[3] = True
                if self.heading_encoded:
                    msk[4] = True
            case 'hsin':
                if self.heading_encoded:
                    msk[3] = True
                else:
                    raise ValueError("Heading is not encoded with sin/cos, cannot get 'hsin' mask")
            case 'hcos':
                if self.heading_encoded:
                    msk[4] = True
                else:
                    raise ValueError("Heading is not encoded with sin/cos, cannot get 'hcos' mask")
            case _:
                raise ValueError(f"Unknown selector '{selector}' for getDataMask")
        #TODO:CHECK check this implementation
        return msk

def split_tr_ev_3to1(dataset:_GDataset)->tuple[_GDataset,_GDataset]:
    total_len = len(dataset)
    train_len = (total_len * 3) // 4
    val_len = total_len - train_len
    train_ds, val_ds = _rsplit(dataset, [train_len, val_len])
    return train_ds, val_ds

def getLbName(label_idx:int,active_labels)->str:
    try:
        return _LE(active_labels[label_idx]).name
    except ValueError:
        return "UNKNOWN_LABEL"

def train_model(model:_tch.nn.Module, train_loader:_GDL, eval_loader:_GDL, epochs:int=10, lr:float=1e-3, weight_decay:float=1e-5, device:str='cpu', verbose:bool=False, *, progress_logging:Progress_logging_options='clilog', active_labels, neg_over_pos_ratio:float=1.0):
    model = model.to(device)
    optimizer = _tch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    posw = _tch.tensor(neg_over_pos_ratio, device=device)
    criterion = _tch.nn.BCEWithLogitsLoss(pos_weight=posw)
    tprint = _TabPrint(tab="   ", enabled=(progress_logging=='clilog'))

    act_labels_num = len(active_labels)

    pl_tracc = _np.zeros((act_labels_num,epochs), dtype=_np.float32)
    tot_tracc = _np.zeros((1,epochs), dtype=_np.float32)
    pl_vacc = _np.zeros((act_labels_num,epochs), dtype=_np.float32)
    tot_vacc = _np.zeros((1,epochs), dtype=_np.float32)

    for epoch in _tqdm(range(epochs), desc="Training Epochs", disable=(progress_logging!='tqdm')):
        tprint(f"\n{_Back.CYAN}{_Fore.YELLOW} ---------- Epoch {epoch+1}/{epochs} ---------- {_Style.RESET_ALL}")
        with tprint.tab:
            tprint(f"{_Fore.YELLOW}{_Style.BRIGHT}--> Training...   {_Style.RESET_ALL}")
            with tprint.tab:
                model.train()
                train_total_loss = 0
                tot_mlb = 0
                tot_correct = _tch.zeros((1,act_labels_num), device=device, dtype=_tch.long)
                # TODO decide whether to use tqdm here or not
                for i,batch in enumerate(_tqdm(train_loader, desc="Training Batches", leave=False)):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    logits = model(batch)
                    scores = _tch.sigmoid(logits)
                    y = batch.y.float().view(batch.num_graphs, act_labels_num)
                    train_loss = criterion(logits, y)
                    train_loss.backward()
                    # TODO remove dbg print
                    if i==len(train_loader)-1:
                        model.printGradInfo()
                        # print num of positive and negative logits in last batch
                        pos_logits = (logits >= 0).long().sum().item()
                        neg_logits = (logits < 0).long().sum().item()
                        tprint(f"Last Batch Logits: Pos={pos_logits}, Neg={neg_logits}")
                    train_total_loss += train_loss.item() * batch.num_graphs
                    optimizer.step()

                    # Accuracy con threshold 0.5
                    preds = (scores >= 0.5).float()
                    corr = (preds == y).long().sum(dim=0)
                    tot_correct += corr
                    tot_mlb += batch.num_graphs
                    acc = corr.sum().item() / (batch.num_graphs * act_labels_num)
                    if verbose:
                        tprint(f"{_Style.DIM}Training Batch Loss: {train_loss.item():.4f}, Training Batch Accuracy: {acc:.4f}{_Style.RESET_ALL}")
            avg_train_loss = train_total_loss / len(train_loader)
            if verbose:
                tprint(f"Training Loss: {avg_train_loss:.4f}")
            tot_train_accuracy = tot_correct.sum().item() / (tot_mlb * act_labels_num)
            per_label_train_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
            tprint(f"{_Fore.GREEN}{_Style.BRIGHT}Training Accuracy: {tot_train_accuracy:.4f}{_Style.RESET_ALL}")
            if verbose:
                tprint(f"Per-Label Training Accuracy:")
                with tprint.tab:
                    for i, acc in enumerate(per_label_train_acc):
                        tprint(f'label "{getLbName(i, active_labels)}" -> {acc:.4f}')

            # Evaluation
            tprint(f"{_Fore.YELLOW}{_Style.BRIGHT}--> Validating ...   {_Style.RESET_ALL}")
            with tprint.tab:
                model.eval()
                val_total_loss = 0
                tot_mlb = 0
                tot_correct = _tch.zeros((1,act_labels_num), device=device, dtype=_tch.long)

                with _tch.no_grad():
                    for batch in _tqdm(eval_loader, desc="Validation Batches", leave=False):
                        batch = batch.to(device)
                        logits = model(batch)
                        scores = _tch.sigmoid(logits)
                        y = batch.y.float().view(batch.num_graphs, act_labels_num)
                        val_loss = criterion(logits, y)
                        val_total_loss += val_loss.item() * batch.num_graphs

                        # Accuracy con threshold 0.5
                        preds = (scores >= 0.5).float()
                        tot_correct += (preds == y).long().sum(dim=0)
                        tot_mlb += batch.num_graphs
            avg_val_loss = val_total_loss / len(eval_loader)
            if verbose:
                tprint(f"Validation Loss: {avg_val_loss:.4f}")
            tot_val_accuracy = tot_correct.sum().item() / (tot_mlb * act_labels_num)
            per_label_val_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
            tprint(f"{_Fore.GREEN}{_Style.BRIGHT}Validation Accuracy: {tot_val_accuracy:.4f}{_Style.RESET_ALL}")
            if verbose:
                tprint(f"Per-Label Eval Accuracy:")
                with tprint.tab:
                    for i, acc in enumerate(per_label_val_acc):
                        tprint(f'label "{getLbName(i, active_labels)}" -> {acc:.4f}')
        pl_tracc[:,epoch] = _np.array(per_label_train_acc)
        pl_vacc[:,epoch] = _np.array(per_label_val_acc)
        tot_tracc[:,epoch] = tot_train_accuracy
        tot_vacc[:,epoch] = tot_val_accuracy
    return (pl_tracc, tot_tracc), (pl_vacc, tot_vacc)

def dataFlattener(data):
    """
    Flattens temporal dimension, bringing samples from shape [vehicles/nodes/batches, frames, features] to [vehicles/nodes/batches, frames*features]
    """ 
    #TODO - try using this function
    data.x = data.x.flatten(start_dim=1)
    return data