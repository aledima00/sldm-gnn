import torch
import torch.multiprocessing as tmp
import optuna
import traceback
from optuna.trial import TrialState
from torch_geometric.loader import DataLoader as GDL
import torch_geometric.transforms as T
from pathlib import Path
import colorama
from colorama import Fore, Style, Back
import numpy as np
from matplotlib import pyplot as plt
import click
import re
import time
from datetime import datetime
from tqdm.auto import tqdm

from src.dataset import MapGraph
from src.models.grusage import GruSage
import src.transforms as TFs
from src.utils import train_model, MetaData
colorama.init(autoreset=True)

EPOCHS=100
EMB_DIM=8
NUM_POSSIBLE_STATION_TYPES=256

def suggest_params(trial: optuna.Trial) -> dict:
    """Define-by-run suggestion of the model/training hyperparameters."""
    combDict = {}

    # ---- Training params ----
    combDict['epochs'] = EPOCHS
    combDict['batch_size'] = trial.suggest_categorical('batch_size', [16, 32, 64])
    combDict['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    combDict['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # ---- Data augmentation ----
    combDict['tf_pos_noise'] = trial.suggest_categorical('tf_pos_noise', [True, False])
    if combDict['tf_pos_noise']:
        combDict['pos_noise_prop_to_speed'] = trial.suggest_categorical('pos_noise_prop_to_speed', [True, False])
        if combDict['pos_noise_prop_to_speed']:
            combDict['pos_noise_std_max'] = trial.suggest_float('pos_noise_std_max', 0.05, 0.5)
            combDict['pos_noise_std'] = 0.2  # unused but kept for getParams compatibility
        else:
            combDict['pos_noise_std'] = trial.suggest_float('pos_noise_std', 0.05, 0.5)
            combDict['pos_noise_std_max'] = 0.2  # unused

    # ---- Loss ----
    combDict['focal_gamma'] = trial.suggest_categorical('focal_gamma', [0.0, 1.0, 2.0])
    if combDict['focal_gamma'] > 0:
        combDict['focal_alpha'] = trial.suggest_categorical('focal_alpha', [None, 0.25, 0.5, 0.75])
    else:
        combDict['focal_alpha'] = None

    # ---- Model architecture ----
    combDict['emb_dim'] = EMB_DIM
    combDict['num_possible_station_types'] = NUM_POSSIBLE_STATION_TYPES

    combDict['gs_dropout'] = trial.suggest_float('gs_dropout', 0.0, 0.5)
    combDict['gs_neg_slope'] = trial.suggest_float('gs_neg_slope', 0.01, 0.3)

    gs_hidden_size = trial.suggest_int('gs_hidden_size', 32, 128, step=32)
    combDict['gs_hidden_size'] = gs_hidden_size
    combDict['gs_gru_hidden_size'] = gs_hidden_size
    combDict['gs_gru_num_layers'] = trial.suggest_int('gs_gru_num_layers', 1, 3)
    combDict['gs_fc1_dims'] = [gs_hidden_size]
    combDict['gs_sage_hidden_dims'] = [gs_hidden_size, gs_hidden_size]
    combDict['gs_pooling'] = trial.suggest_categorical('gs_pooling', ['double', 'mean', 'max', 'sum'])
    combDict['gs_fc2_dims'] = [gs_hidden_size // 3]

    gs_map_hidden_size = trial.suggest_int('gs_map_hidden_size', 16, 64, step=16)
    combDict['gs_map_hidden_size'] = gs_map_hidden_size
    combDict['gs_mapenc_lane_embdim'] = gs_map_hidden_size // 4
    combDict['gs_mapenc_sage_hdims'] = [gs_map_hidden_size, gs_map_hidden_size]
    combDict['gs_map_attention_topk'] = trial.suggest_int('gs_map_attention_topk', 3, 10)

    return combDict


# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"{Fore.CYAN}Using device: {DEVICE}{Style.RESET_ALL}")

def getConfigDir(study_root: Path, config_index: int) -> Path:
    cfg = study_root / f"config{config_index:02d}"
    cfg.mkdir(parents=True, exist_ok=True)
    return cfg


def getParams(bin_stats: tuple | None, tot_vacc: np.ndarray, cut: int | None = None, *, combDict: dict) -> str:
    """Parameters as string for plot text box"""
    EMB_DIM = combDict.get('emb_dim')
    EPOCHS = combDict.get('epochs')
    BATCH_SIZE = combDict.get('batch_size')
    LR = combDict.get('lr')
    WEIGHT_DECAY = combDict.get('weight_decay')
    GS_GRU_HS = combDict.get('gs_gru_hidden_size')
    GS_GRU_NL = combDict.get('gs_gru_num_layers')
    GS_FC1_DIMS = combDict.get('gs_fc1_dims')
    GS_SAGE_HIDDEN_DIMS = combDict.get('gs_sage_hidden_dims')
    GS_FC2_DIMS = combDict.get('gs_fc2_dims')
    GS_DROPOUT = combDict.get('gs_dropout')
    GS_NEGSLOPE = combDict.get('gs_neg_slope')
    GS_MELD = combDict.get('gs_mapenc_lane_embdim')
    GS_MESD = combDict.get('gs_mapenc_sage_hdims')
    GS_MAPATTENTION_TOPK = combDict.get('gs_map_attention_topk')
    params = (f"GRUSAGE model parameters:\n"
              f" - Embedding size for station types: {EMB_DIM}\n"
              f" - GRU: hidden size = {GS_GRU_HS}, num layers = {GS_GRU_NL}\n"
              f" - FC1 dims: {GS_FC1_DIMS}\n"
              f" - SAGE hidden dims: {GS_SAGE_HIDDEN_DIMS}\n"
              f" - FC2 dims: {GS_FC2_DIMS}\n"
              f" - Regularization: Dropout = {GS_DROPOUT}, ReLU Neg. slope = {GS_NEGSLOPE}\n"
              f"Map Input processing:\n"
              f" - Map Encoder: Lane emb.dim = {GS_MELD}, Sage HDims = {GS_MESD}\n"
              f" - Map Spatial Attn: topk = {GS_MAPATTENTION_TOPK}\n")
    params += "\n"
    params += f"Tr. Params: EP: {EPOCHS}, BS: {BATCH_SIZE}, LR: {LR}, WD: {WEIGHT_DECAY}\n"
    FOCAL_GAMMA = combDict.get('focal_gamma')
    if FOCAL_GAMMA and FOCAL_GAMMA > 0:
        FOCAL_ALPHA = combDict.get('focal_alpha')
        params += f"Loss: Focal (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})\n"
    else:
        params += "Loss: BCEWithLogits (pos_weight=neg/pos ratio)\n"
    params += "Data Augmentation:\n"
    if combDict.get('tf_pos_noise'):
        if combDict.get('pos_noise_prop_to_speed'):
            POS_NOISE_STD_MAX = combDict.get('pos_noise_std_max')
            params += f" - Add Noise on Positions (X,Y) prop to speed, with max std: {POS_NOISE_STD_MAX}\n"
        else:
            POS_NOISE_STD = combDict.get('pos_noise_std')
            params += f" - Add Noise on Positions (X,Y) with std: {POS_NOISE_STD}\n"
    if cut is not None:
        params += f" - Cutting after: {cut} frames\n"

    best_vacc_idx = tot_vacc[0, :].argmax()
    best_vacc = tot_vacc[0, best_vacc_idx]
    params += f"\nBest Val. Acc.: {best_vacc:.4f}, @ep.{best_vacc_idx}\n"

    if bin_stats is not None:
        (bin_cm_flat_values, bin_rocauc_values) = bin_stats

        best_rocauc_idx = bin_rocauc_values[0, :].argmax()
        best_rocauc = bin_rocauc_values[0, best_rocauc_idx]
        params += f"Best Val. ROC AUC: {best_rocauc:.4f}, @ep.{best_rocauc_idx}\n"
    return params

def _move_mu_sigma(mu_sigma, device):
    """Move a (mu, sigma) tuple of dicts to the given device."""
    mu, sigma = mu_sigma
    return (
        {k: v.to(device) for k, v in mu.items()},
        {k: v.to(device) for k, v in sigma.items()},
    )




def build_and_train(trial: optuna.Trial, *, inputdir: Path, study_root: Path, lbnum: int, cut: int | None, include_map: bool, mu_sigma_cpu, quiet: bool, epoch_progress_q, epoch_callback) -> float:
    """Build the model + data for one trial, train it, save plots/state, return best val accuracy."""
    combDict = suggest_params(trial)

    inpath = inputdir.resolve()

    cfgdir = getConfigDir(study_root, trial.number)
    fbase = f"GRUSAGE_{'MAP_' if include_map else ''}"
    plot_fname = f"{fbase}_trev_plot.png"
    state_fname = f"{fbase}_best_state.pth"

    tr_gpath = inpath / 'train' / '.graphs'
    ev_gpath = inpath / 'eval' / '.graphs'
    tr_metadata = MetaData.loadJson(tr_gpath / 'metadata.json')
    ev_metadata = MetaData.loadJson(ev_gpath / 'metadata.json')

    transform = []
    if combDict.get('tf_pos_noise'):
        posnoisestd = combDict.get('pos_noise_std')
        posnoisestdmax = combDict.get('pos_noise_std_max')
        posnoiseproptospeed = combDict.get('pos_noise_prop_to_speed')
        transform.append(TFs.AddNoise(
            target='pos',
            std=posnoisestdmax if posnoiseproptospeed else posnoisestd,
            prop_to_speed=posnoiseproptospeed,
            metadata=tr_metadata,
        ))
    if cut is not None:
        transform.append(TFs.CutFrames(cut))
    transform = T.Compose(transform)

    mu_sigma = _move_mu_sigma(mu_sigma_cpu, DEVICE)
    d_train = MapGraph(tr_gpath, device=DEVICE, transform=transform, normalizeZScore=True, metadata=tr_metadata, zscore_mu_sigma=mu_sigma)
    d_eval = MapGraph(ev_gpath, device=DEVICE, transform=transform, normalizeZScore=True, metadata=ev_metadata, zscore_mu_sigma=mu_sigma)

    # create dataloaders
    dl_train = GDL(d_train, batch_size=combDict.get('batch_size'), shuffle=True)
    dl_eval = GDL(d_eval, batch_size=combDict.get('batch_size'), shuffle=True)

    # load map data if required
    if include_map:
        map_path = inpath / '.map' / 'vmap.pth'
        map_tensors = torch.load(map_path, map_location=DEVICE)
    else:
        map_tensors = None

    model = GruSage(
        dynamic_features_num=tr_metadata.n_node_temporal_features,
        frames_num=tr_metadata.frames_num,
        gru_hidden_size=combDict.get('gs_gru_hidden_size'),
        gru_num_layers=combDict.get('gs_gru_num_layers'),
        fc1dims=combDict.get('gs_fc1_dims'),
        sage_hidden_dims=combDict.get('gs_sage_hidden_dims'),
        fc2dims=combDict.get('gs_fc2_dims'),
        out_dim=len(tr_metadata.active_labels),
        num_st_types=combDict.get('num_possible_station_types'),
        emb_dim=combDict.get('emb_dim'),
        dropout=combDict.get('gs_dropout'),
        negative_slope=combDict.get('gs_neg_slope'),
        global_pooling=combDict.get('gs_pooling'),
        map_tensors=map_tensors,
        mapenc_lane_embdim=combDict.get('gs_mapenc_lane_embdim'),
        mapenc_sage_hdims=combDict.get('gs_mapenc_sage_hdims'),
        map_attention_topk=combDict.get('gs_map_attention_topk'),
    )

    mu_sigma_dict = {'mu': mu_sigma_cpu[0], 'sigma': mu_sigma_cpu[1]}
    (tot_tracc, tot_vacc, bin_stats) = runModel(
        model, tr_metadata, dl_train, dl_eval,
        combDict=combDict,
        best_state_outfile=cfgdir / state_fname,
        norm_stats_dict_for_snapshot=mu_sigma_dict,
        quiet=quiet,
        epoch_progress_q=epoch_progress_q,
        epoch_callback=epoch_callback,
    )
    plotAccuracies(tot_tracc, tot_vacc, bin_stats, cfgdir / plot_fname, lbnum, cut=cut, combDict=combDict)

    best_vacc = float(tot_vacc[0, :].max())
    return best_vacc


def _objective_factory(shared_args):
    """Return a closure capturing shared (read-only) args, suitable as optuna objective."""
    def objective(trial: optuna.Trial) -> float:
        def epoch_callback(epoch, vacc):
            trial.report(vacc, epoch)
            if trial.should_prune():
                tqdm.write(f"{Fore.YELLOW}Trial {trial.number} pruned at epoch {epoch+1} (vacc={vacc:.4f}){Style.RESET_ALL}")
                raise optuna.TrialPruned()

        try:
            return build_and_train(trial, **shared_args, epoch_callback=epoch_callback)
        except optuna.TrialPruned:
            raise
        except Exception as e:
            tqdm.write(f"{Fore.RED}Trial {trial.number} FAILED: {e}{Style.RESET_ALL}")
            tqdm.write(traceback.format_exc())
            raise
    return objective


def _worker(worker_id: int, n_trials_this_worker: int, study_name: str, storage: str, shared_args: dict):
    """Process entrypoint: attach to the shared study and run n_trials_this_worker trials."""
    try:
        tmp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    study = optuna.load_study(study_name=study_name, storage=storage)
    objective = _objective_factory(shared_args)
    study.optimize(objective, n_trials=n_trials_this_worker, n_jobs=1, catch=(Exception,), show_progress_bar=False)


@click.command()
@click.argument('inputdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.argument('outdir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.option('-l', '--label-num', 'lbnum', type=int, required=True, prompt='Label number to train the model on')
@click.option('--cut', type=int, default=None, help='If set, cuts frames after the given number, allowing prediction at earlier timesteps')
@click.option('--include-map', is_flag=True, default=False, help='If set, includes map information as node features (if available in dataset)')
@click.option('-n', '--n-trials', 'n_trials', type=int, default=32, show_default=True, help='Total number of Optuna trials to run (across all workers).')
@click.option('-T', '--threads', 'n_threads', type=int, default=1, show_default=True, help='Number of parallel worker processes. Each runs Optuna trials on the GPU; keep small to avoid CUDA OOM.')
@click.option('--study-name', 'study_name', type=str, default=None, help='Optuna study name. Defaults to "sweep_<outdir name>".')
def main(inputdir: Path, outdir: Path, lbnum: int, cut: int | None, include_map: bool, n_trials: int, n_threads: int, study_name: str | None):
    outdir.mkdir(parents=True, exist_ok=True)
    if study_name is None:
        ts = datetime.now().strftime('%Y%m%d-%H%M')
        study_name = f"sweep_{outdir.name}_{ts}"
    study_root = outdir / study_name
    study_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{(study_root / 'sweep.db').resolve()}"

    # startup trials: unpruned first trials to create a baseline stats for pruning
    # warmup steps: number of epochs to run before pruning is allowed
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"{Fore.CYAN}Optuna study: {study_name}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Storage: {storage}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Trials requested: {n_trials}, workers: {n_threads}{Style.RESET_ALL}")
    n_existing = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)])
    n_remaining = max(0, n_trials - n_existing)
    if n_existing > 0:
        print(f"{Fore.CYAN}Found {n_existing} existing trials; {n_remaining} remaining{Style.RESET_ALL}")
    if n_remaining == 0:
        print(f"{Fore.GREEN}Study already reached {n_trials} trials. Nothing to do.{Style.RESET_ALL}")
        _print_best(study)
        return
    if not click.confirm("Proceed?", default=True):
        return

    # ---- Precompute z-score (mu, sigma) once on the parent (GPU for speed, CPU for handoff) ----
    inpath = inputdir.resolve()
    tr_gpath = inpath / 'train' / '.graphs'
    tr_metadata = MetaData.loadJson(tr_gpath / 'metadata.json')
    print(f"{Fore.CYAN}Precomputing dataset mu/sigma on {DEVICE} (shared across all trials){Style.RESET_ALL}")
    _tmp_ds = MapGraph(tr_gpath, device=DEVICE, transform=None, normalizeZScore=True, metadata=tr_metadata)
    mu_sigma_gpu = _tmp_ds.getMuSigma()
    mu_sigma_cpu = _move_mu_sigma(mu_sigma_gpu, 'cpu')
    del _tmp_ds
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    shared_args = dict(
        inputdir=inputdir, study_root=study_root, lbnum=lbnum, cut=cut, include_map=include_map,
        mu_sigma_cpu=mu_sigma_cpu, quiet=(n_threads > 1), epoch_progress_q=None,
    )

    if n_threads <= 1:
        # ---- Sequential: run in the parent process ----
        objective = _objective_factory(shared_args)
        with tqdm(total=n_remaining, desc='Trials') as pbar:
            def callback(study, trial):
                pbar.update(1)
            study.optimize(objective, n_trials=n_remaining, n_jobs=1, callbacks=[callback], catch=(Exception,), show_progress_bar=False)
    else:
        # ---- Parallel: spawn T worker processes, each consuming the shared study ----
        ctx = tmp.get_context('spawn')
        # Distribute n_remaining trials as evenly as possible across workers
        base = n_remaining // n_threads
        rem = n_remaining % n_threads
        quotas = [base + (1 if i < rem else 0) for i in range(n_threads)]

        procs = []
        for wid, q in enumerate(quotas):
            if q <= 0:
                continue
            p = ctx.Process(target=_worker, args=(wid, q, study_name, storage, shared_args))
            procs.append(p)

        print(f"{Fore.CYAN}Launching {len(procs)} parallel workers (quotas={quotas}){Style.RESET_ALL}")
        for p in procs:
            p.start()

        # Father: poll the study DB and show a global progress bar over completed/pruned trials
        with tqdm(total=n_remaining, desc='Trials') as pbar:
            prev_done = 0
            while any(p.is_alive() for p in procs):
                time.sleep(1.0)
                # refresh from DB
                cur_done = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]) - n_existing
                if cur_done > prev_done:
                    pbar.update(cur_done - prev_done)
                    prev_done = cur_done
            # final drain
            cur_done = len([t for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.PRUNED)]) - n_existing
            if cur_done > prev_done:
                pbar.update(cur_done - prev_done)

        for p in procs:
            p.join()

    print()
    _print_best(study)


def _print_best(study: optuna.Study):
    """Print the best trial summary."""
    try:
        best = study.best_trial
    except ValueError:
        print(f"{Fore.YELLOW}No completed trials to report.{Style.RESET_ALL}")
        return
    print(f"{Fore.GREEN}{Style.BRIGHT}Best trial #{best.number}: val_acc={best.value:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Params:{Style.RESET_ALL}")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"{Fore.CYAN}Trials summary: "
          f"COMPLETE={len([t for t in study.trials if t.state==TrialState.COMPLETE])}, "
          f"PRUNED={len([t for t in study.trials if t.state==TrialState.PRUNED])}, "
          f"FAIL={len([t for t in study.trials if t.state==TrialState.FAIL])}{Style.RESET_ALL}")



def plotAccuracies(tot_tracc: np.ndarray, tot_vacc: np.ndarray, bin_stats: tuple | None, outfile: Path, lbnum: int, *, cut, combDict: dict):
    fig, (ax_plot, ax_text) = plt.subplots(
        1, 2,
        figsize=(10, 4),
        gridspec_kw={'width_ratios': [3, 2]}
    )
    plt_yticks = np.arange(-0.1, 1.2, 0.1)
    ax_plot.plot(tot_vacc[0, :], color='blue', label='Val. Acc.')
    ax_plot.plot(tot_tracc[0, :], color='orange', linestyle='--', label='Tr. Acc.')

    if bin_stats is not None:
        (bin_cm_flat_values, bin_rocauc_values) = bin_stats
        tn = bin_cm_flat_values[0, :]
        fp = bin_cm_flat_values[1, :]
        fn = bin_cm_flat_values[2, :]
        tp = bin_cm_flat_values[3, :]
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        ax_plot.plot(bin_rocauc_values[0, :], color='purple', label='Val. ROC AUC')
        ax_plot.plot(precision, color='green', alpha=0.2, label='Val. Precision')
        ax_plot.plot(recall, color='red', alpha=0.2, label='Val. Recall')

    ax_plot.set_ylim(bottom=0, top=1)
    ax_plot.set_yticks(plt_yticks)
    ax_plot.grid(True)
    ax_plot.legend()
    ax_plot.set_title(f'Validation Accuracy for label #{lbnum}')
    
    # text box with final results
    params_text = getParams(bin_stats, tot_vacc, cut=cut, combDict=combDict)
    ax_text.axis('off')
    ax_text.text(0,0.95, params_text, va='top')

    fig.tight_layout()
    plt.savefig(outfile)
    plt.close(fig)
def runModel(model, train_metadata: MetaData, dl_train, dl_eval, *, combDict: dict, best_state_outfile: Path | None = None, norm_stats_dict_for_snapshot: dict | None = None, quiet: bool = False, epoch_progress_q=None, epoch_callback=None):
    train_prior = train_metadata.n_positive / train_metadata.n_samples
    (_, tot_tracc), (_, tot_vacc), bin_stats = train_model(
        model,
        dl_train,
        dl_eval,
        epochs=combDict.get('epochs'),
        lr=combDict.get('lr'),
        weight_decay=combDict.get('weight_decay'),
        device=DEVICE,
        active_labels=train_metadata.active_labels,
        neg_over_pos_ratio=train_metadata.getNegOverPosRatio(),
        best_state_path=best_state_outfile,
        norm_stats_dict_for_snapshot=norm_stats_dict_for_snapshot,
        train_prior=train_prior,
        focal_alpha=combDict.get('focal_alpha'),
        focal_gamma=combDict.get('focal_gamma'),
        epoch_progress_q=epoch_progress_q,
        quiet=quiet,
        epoch_callback=epoch_callback,
    )
    return (tot_tracc, tot_vacc, bin_stats)

if __name__ == '__main__':
    main()
