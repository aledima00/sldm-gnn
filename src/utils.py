import torch as _tch
from torch_geometric.loader import DataLoader as _GDL
from typing import Literal as _Lit, Iterable as _Iterable, Any as _Any, List as _List, Tuple as _Tuple, Callable as _Callable, TypeAlias as _TA, TypedDict as _TypedDict
from colorama import Fore as _Fore, Back as _Back, Style as _Style
from tqdm.auto import tqdm as _tqdm
import numpy as _np
from dataclasses import dataclass as _dc
from pathlib import Path as _Path
import json as _json
from sklearn.metrics import confusion_matrix as _confmat, roc_auc_score as _rocauc
from itertools import product as _iproduct

from .tprint import TabPrint as _TabPrint
from .labels import LabelsEnum as _LE
from .models.grusage import GruSage as _Grusage

Progress_logging_options = _Lit['clilog', 'tqdm', 'none']

FmaskType = _Lit['x','y','pos','speed','heading','hsin','hcos']


callTuple: _TA = _Tuple[_Callable, str]


def saveSnapshot(model:_Grusage, path:_Path, *, norm_stats_dict:dict|None=None):
    snapshot_dict = {
        'state_dict': model.state_dict_no_mapenc(),
        'ip_dict': model.input_params_dict(),
        'norm_stat_dict': norm_stats_dict
    }
    _tch.save(snapshot_dict, path.resolve())

class SnapshotDict(_TypedDict):
    state_dict: dict
    ip_dict: dict
    norm_stat_dict: dict|None

def loadSnapshot(path:_Path)->SnapshotDict:
    pr = path.resolve()
    assert pr.exists() and pr.is_file(), f"Snapshot file not found at {path}"
    snap = _tch.load(pr)
    assert 'state_dict' in snap and 'ip_dict' in snap, f"Snapshot file at {path} is missing required keys"
    if 'norm_stat_dict' not in snap:
        snap['norm_stat_dict'] = None
    return snap

class ParamSweepContext:

    def __init__(self,params_dict:dict[str, _List|callTuple]):
        """
        
        :param params_dict: a dictionary where keys are parameter names and values are either lists of values to sweep, or tuples of (callable, dependency_param_name) where callable is a function that takes the value of the dependency parameter and returns a list of values to sweep.
        """
        for name, val in params_dict.items():
            assert isinstance(name, str), f"Parameter name must be a string, got {type(name)}"
            assert isinstance(val, (list, tuple)), f"Parameter values must be a list or a (callable, str) tuple, got {type(val)} for parameter '{name}'"
            if isinstance(val, tuple):
                assert len(val) == 2, f"Parameter value tuple must have length 2, got {len(val)} for parameter '{name}'"
                assert callable(val[0]), f"First element of parameter value tuple must be callable, got {type(val[0])} for parameter '{name}'"
                assert isinstance(val[1], str), f"Second element of parameter value tuple must be a string (dependency parameter name), got {type(val[1])} for parameter '{name}'"
        
        
        self._lambdas = {name:val for name,val in params_dict.items() if isinstance(val, tuple)}
        pd = params_dict.copy()
        for name in self._lambdas.keys():
            del pd[name]

        val_keys = list(pd.keys())
        self._params_idx = {name:idx for idx,name in enumerate(val_keys)}
        self._values_list = [params_dict[name] for name in val_keys]
        

    
    # define generator that yields all combinations of parameters, in form of key-value dicts
    def combinations(self)->_Iterable[dict[str, _Any]]:
        """
        Generate the combinations dict of parameter values.
        Yields:
            dict[str, Any]: A dictionary where keys are parameter names and values are the corresponding parameter
        """
        
        # generate all combinations of parameter values
        for cur_comb_tuple in _iproduct(*self._values_list):
            comb_dict = {name:cur_comb_tuple[idx] for name, idx in self._params_idx.items()}
            # evaluate lambdas with current combination
            for name, (func, dep_name) in self._lambdas.items():
                dep_value = comb_dict.get(dep_name)
                if dep_value is None:
                    raise ValueError(f"Dependency parameter '{dep_name}' not found in current combination for parameter '{name}'")
                comb_dict[name] = func(dep_value)
            #print(f"current comb dict:", comb_dict)
            yield comb_dict

    def __len__(self)->int:
        """
        Returns the total number of combinations in the sweep.
        """
        tot = 1
        for vals in self._values_list:
            tot *= len(vals)
        return tot

@_dc
class MetaData:
    n_samples:int
    n_positive:int
    n_edge_features:int
    frames_num:int
    m_radius:float
    active_labels:list[int]

    @property
    def n_node_temporal_features(self)->int:
        return (3 + 1 + 2)

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
    
    def getFeaturesMask(self,selector:FmaskType)->_tch.Tensor:
        msk = _tch.full((self.n_node_temporal_features,),False,dtype=_tch.bool)
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
                msk[4] = True
            case 'hsin':
                msk[3] = True
            case 'hcos':
                msk[4] = True
            case _:
                raise ValueError(f"Unknown selector '{selector}' for getFeaturesMask")
        return msk

def getLbName(label_idx:int,active_labels)->str:
    try:
        return _LE(active_labels[label_idx]).name
    except ValueError:
        return "UNKNOWN_LABEL"

def train_model(model:_Grusage, train_loader:_GDL, eval_loader:_GDL, epochs:int=10, lr:float=1e-3, weight_decay:float=1e-5, device:str='cpu', verbose:bool=False, *, progress_logging:Progress_logging_options='clilog', active_labels, neg_over_pos_ratio:float=1.0, best_state_path:_Path|None=None,norm_stats_dict_for_snapshot:dict|None=None):
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

    best_vacc = 0

    if act_labels_num==1:
        bin_cm_flat_values = _np.zeros((4,epochs), dtype=_np.int32)  # tn,fp,fn,tp
        bin_rocauc_values = _np.zeros((1,epochs), dtype=_np.float32)

    for epoch in _tqdm(range(epochs), desc="Training Epochs", disable=(progress_logging!='tqdm')):
        tprint(f"\n{_Back.CYAN}{_Fore.YELLOW} ---------- Epoch {epoch+1}/{epochs} ---------- {_Style.RESET_ALL}")
        with tprint.tab:
            # ============================ Training ============================
            tprint(f"{_Fore.YELLOW}{_Style.BRIGHT}--> Training...   {_Style.RESET_ALL}")
            with tprint.tab:
                model.train()
                train_total_loss = 0
                tot_mlb = 0
                tot_correct = _tch.zeros((1,act_labels_num), device=device, dtype=_tch.long)
                for i,batch in enumerate(_tqdm(train_loader, desc="Training Batches", leave=False)):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    logits = model(batch)
                    scores = _tch.sigmoid(logits)
                    y = batch.y.float().view(batch.num_graphs, act_labels_num)
                    train_loss = criterion(logits, y)
                    train_loss.backward()
                    if i==len(train_loader)-1:
                        # model.printGradInfo()
                        # print num of positive and negative logits in last batch
                        pos_logits = (logits >= 0).long().sum().item()
                        neg_logits = (logits < 0).long().sum().item()
                        #tprint(f"Last Batch Logits: Pos={pos_logits}, Neg={neg_logits}")
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

            # ============================ Validation ============================
            tprint(f"{_Fore.YELLOW}{_Style.BRIGHT}--> Validating ...   {_Style.RESET_ALL}")
            with tprint.tab:
                model.eval()
                val_total_loss = 0
                tot_mlb = 0
                tot_correct = _tch.zeros((1,act_labels_num), device=device, dtype=_tch.long)
                val_scores = []
                val_preds = []
                val_gt = []
                

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

                        # accuracy metrics
                        tot_correct += (preds == y).long().sum(dim=0)
                        tot_mlb += batch.num_graphs

                        if act_labels_num == 1:
                            # binary metrics collection
                            val_scores.append(scores)
                            val_preds.append(preds)
                            val_gt.append(y)
                        
            avg_val_loss = val_total_loss / len(eval_loader)
            if verbose:
                tprint(f"Validation Loss: {avg_val_loss:.4f}")
            tot_val_accuracy = tot_correct.sum().item() / (tot_mlb * act_labels_num)

            if tot_val_accuracy > best_vacc and best_state_path is not None:
                best_vacc = tot_val_accuracy
                saveSnapshot(model, best_state_path, norm_stats_dict=norm_stats_dict_for_snapshot)
                tprint(f"{_Fore.GREEN}{_Style.BRIGHT}New best model saved with Validation Accuracy: {best_vacc:.4f}{_Style.RESET_ALL}")

            per_label_val_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
            tprint(f"{_Fore.GREEN}{_Style.BRIGHT}Validation Accuracy: {tot_val_accuracy:.4f}{_Style.RESET_ALL}")

            if act_labels_num == 1:
                # in binary tasks, compute confusion matrix and F1-score in each validation epoch (no auc roc here)
                val_scores = _tch.cat(val_scores, dim=0).squeeze().cpu().numpy()
                val_preds = _tch.cat(val_preds, dim=0).squeeze().cpu().numpy()
                val_gt = _tch.cat(val_gt, dim=0).squeeze().cpu().numpy()

                cm = _confmat(val_gt, val_preds)
                tn,fp,fn,tp = cm.ravel()
                tprint(f"Validation Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                roc_auc = _rocauc(val_gt, val_scores)
                tprint(f"{_Fore.MAGENTA}{_Style.BRIGHT}Validation Stats: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}{_Style.RESET_ALL}, ROC AUC={roc_auc:.4f}")
                # store binary metrics
                bin_cm_flat_values[:,epoch] = _np.array([tn,fp,fn,tp], dtype=_np.int32)
                bin_rocauc_values[0,epoch] = roc_auc

            if verbose:
                tprint(f"Per-Label Eval Accuracy:")
                with tprint.tab:
                    for i, acc in enumerate(per_label_val_acc):
                        tprint(f'label "{getLbName(i, active_labels)}" -> {acc:.4f}')
        pl_tracc[:,epoch] = _np.array(per_label_train_acc)
        pl_vacc[:,epoch] = _np.array(per_label_val_acc)
        tot_tracc[:,epoch] = tot_train_accuracy
        tot_vacc[:,epoch] = tot_val_accuracy

    return (pl_tracc, tot_tracc), (pl_vacc, tot_vacc), ((bin_cm_flat_values, bin_rocauc_values) if act_labels_num==1 else None)