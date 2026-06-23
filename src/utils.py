import torch as _tch
from torch_geometric.loader import DataLoader as _GDL
from typing import Literal as _Lit, Iterable as _Iterable, Any as _Any, List as _List, Tuple as _Tuple, Callable as _Callable, TypeAlias as _TA, TypedDict as _TypedDict
from colorama import Fore as _Fore, Style as _Style
from tqdm.auto import tqdm as _tqdm
import numpy as _np
from dataclasses import dataclass as _dc
from pathlib import Path as _Path
import json as _json
from sklearn.metrics import confusion_matrix as _confmat, roc_auc_score as _rocauc
from itertools import product as _iproduct
from multiprocessing.sharedctypes import Synchronized as _Synchronized

from .labels import LabelsEnum as _LE
from .models.grusage import GruSage as _Grusage

FmaskType = _Lit['x','y','pos','speed','heading','hsin','hcos']


callTuple: _TA = _Tuple[_Callable, str]


def saveSnapshot(model:_Grusage, path:_Path, *, norm_stats_dict:dict|None=None, train_prior:float|None=None, loss_info:dict|None=None):
    snapshot_dict = {
        'state_dict': model.state_dict_no_mapenc(),
        'ip_dict': model.input_params_dict(),
        'norm_stat_dict': norm_stats_dict,
        'train_prior': train_prior,
        'loss_info': loss_info
    }
    _tch.save(snapshot_dict, path.resolve())

class SnapshotDict(_TypedDict):
    state_dict: dict
    ip_dict: dict
    norm_stat_dict: dict|None
    train_prior: float|None
    loss_info: dict|None

def loadSnapshot(path:_Path)->SnapshotDict:
    pr = path.resolve()
    assert pr.exists() and pr.is_file(), f"Snapshot file not found at {path}"
    snap = _tch.load(pr)
    assert 'state_dict' in snap and 'ip_dict' in snap, f"Snapshot file at {path} is missing required keys"
    if 'norm_stat_dict' not in snap:
        snap['norm_stat_dict'] = None
    if 'train_prior' not in snap:
        snap['train_prior'] = None
    if 'loss_info' not in snap:
        snap['loss_info'] = None
    return snap

def bayesPriorShift(scores, train_prior:float, test_prior:float):
    train_neg = 1.0 - train_prior
    test_neg = 1.0 - test_prior
    prior_ratio = (test_prior / test_neg) / (train_prior / train_neg)
    return scores * prior_ratio / (scores * prior_ratio + (1.0 - scores)), prior_ratio

def focal_bce_loss(logits, targets, alpha=0.75, gamma=2.0):
    bce = _tch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = _tch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()

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

def getLbName(lb_value)->str:
    try:
        return _LE(lb_value).name
    except ValueError:
        return "UNKNOWN_LABEL"

def train_model(model:_Grusage, train_loader:_GDL, eval_loader:_GDL, epochs:int=10, lr:float=1e-3, weight_decay:float=1e-5, device:str='cpu', *, active_labels, neg_over_pos_ratio:float=1.0, best_state_path:_Path|None=None, norm_stats_dict_for_snapshot:dict|None=None, train_prior:float|None=None, focal_alpha:float|None=None, focal_gamma:float=0.0, epoch_progress_counter:_Synchronized|None=None, quiet:bool=False):
    model = model.to(device)
    optimizer = _tch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if focal_gamma > 0:
        if focal_alpha is None:
            neg_frac = neg_over_pos_ratio / (1 + neg_over_pos_ratio)
            focal_alpha = neg_frac
        use_focal = True
        loss_info = {'type': 'focal', 'alpha': focal_alpha, 'gamma': focal_gamma}
    else:
        posw = _tch.tensor(neg_over_pos_ratio, device=device)
        criterion = _tch.nn.BCEWithLogitsLoss(pos_weight=posw)
        use_focal = False
        loss_info = {'type': 'BCEWithLogits', 'pos_weight': float(neg_over_pos_ratio)}

    def compute_loss(logits, y):
        if use_focal:
            return focal_bce_loss(logits, y, alpha=focal_alpha, gamma=focal_gamma)
        else:
            return criterion(logits, y)

    act_labels_num = len(active_labels)

    pl_tracc = _np.zeros((act_labels_num,epochs), dtype=_np.float32)
    tot_tracc = _np.zeros((1,epochs), dtype=_np.float32)
    pl_vacc = _np.zeros((act_labels_num,epochs), dtype=_np.float32)
    tot_vacc = _np.zeros((1,epochs), dtype=_np.float32)

    best_vacc = 0

    if act_labels_num==1:
        bin_cm_flat_values = _np.zeros((4,epochs), dtype=_np.int32)  # tn,fp,fn,tp
        bin_rocauc_values = _np.zeros((1,epochs), dtype=_np.float32)

    epoch_pbar = _tqdm(range(epochs), desc="Training Epochs", disable=quiet)
    for epoch in epoch_pbar:
        # ============================ Training ============================
        model.train()
        train_total_loss = 0
        tot_mlb = 0
        tot_correct = _tch.zeros((1,act_labels_num), device=device, dtype=_tch.long)
        for i,batch in enumerate(_tqdm(train_loader, desc="Training Batches", leave=False, disable=quiet)):
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            scores = _tch.sigmoid(logits)
            y = batch.y.float().view(batch.num_graphs, act_labels_num)
            train_loss = compute_loss(logits, y)
            train_loss.backward()
            train_total_loss += train_loss.item() * batch.num_graphs
            optimizer.step()

            # Accuracy con threshold 0.5
            preds = (scores >= 0.5).float()
            corr = (preds == y).long().sum(dim=0)
            tot_correct += corr
            tot_mlb += batch.num_graphs
        avg_train_loss = train_total_loss / len(train_loader)
        tot_train_accuracy = tot_correct.sum().item() / (tot_mlb * act_labels_num)
        per_label_train_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
        epoch_pbar.set_postfix(tr_loss=f"{avg_train_loss:.4f}", tr_acc=f"{tot_train_accuracy:.4f}")

        # ============================ Validation ============================
        model.eval()
        val_total_loss = 0
        tot_mlb = 0
        tot_correct = _tch.zeros((1,act_labels_num), device=device, dtype=_tch.long)
        val_scores = []
        val_preds = []
        val_gt = []

        with _tch.no_grad():
            for batch in _tqdm(eval_loader, desc="Validation Batches", leave=False, disable=quiet):
                batch = batch.to(device)
                logits = model(batch)
                scores = _tch.sigmoid(logits)
                y = batch.y.float().view(batch.num_graphs, act_labels_num)
                val_loss = compute_loss(logits, y)
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
        tot_val_accuracy = tot_correct.sum().item() / (tot_mlb * act_labels_num)

        if tot_val_accuracy > best_vacc and best_state_path is not None:
            best_vacc = tot_val_accuracy
            saveSnapshot(model, best_state_path, norm_stats_dict=norm_stats_dict_for_snapshot, train_prior=train_prior, loss_info=loss_info)
            if not quiet:
                _tqdm.write(f"{_Fore.GREEN}{_Style.BRIGHT}New best model saved with Validation Accuracy: {best_vacc:.4f}{_Style.RESET_ALL}")

        per_label_val_acc = (tot_correct.sum(dim=0).cpu().float().numpy() / tot_mlb).tolist()
        epoch_pbar.set_postfix(tr_loss=f"{avg_train_loss:.4f}", tr_acc=f"{tot_train_accuracy:.4f}", vl_loss=f"{avg_val_loss:.4f}", vl_acc=f"{tot_val_accuracy:.4f}")

        if act_labels_num == 1:
            # in binary tasks, compute confusion matrix and F1-score in each validation epoch (no auc roc here)
            val_scores = _tch.cat(val_scores, dim=0).squeeze().cpu().numpy()
            val_preds = _tch.cat(val_preds, dim=0).squeeze().cpu().numpy()
            val_gt = _tch.cat(val_gt, dim=0).squeeze().cpu().numpy()

            cm = _confmat(val_gt, val_preds)
            tn,fp,fn,tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            roc_auc = _rocauc(val_gt, val_scores)
            if not quiet:
                _tqdm.write(f"{_Fore.MAGENTA}{_Style.BRIGHT}Epoch {epoch+1}: Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}, ROC AUC={roc_auc:.4f}, CM(TP={tp},TN={tn},FP={fp},FN={fn}){_Style.RESET_ALL}")
            # store binary metrics
            bin_cm_flat_values[:,epoch] = _np.array([tn,fp,fn,tp], dtype=_np.int32)
            bin_rocauc_values[0,epoch] = roc_auc

        pl_tracc[:,epoch] = _np.array(per_label_train_acc)
        pl_vacc[:,epoch] = _np.array(per_label_val_acc)
        tot_tracc[:,epoch] = tot_train_accuracy
        tot_vacc[:,epoch] = tot_val_accuracy

        if epoch_progress_counter is not None:
            with epoch_progress_counter.get_lock():
                epoch_progress_counter.value += 1

    return (pl_tracc, tot_tracc), (pl_vacc, tot_vacc), ((bin_cm_flat_values, bin_rocauc_values) if act_labels_num==1 else None)