import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader as GDL

import src.transforms as TFs
from src.dataset import MapGraph
from src.models.grusage import GruSage
from src.utils import MetaData, bayesPriorShift

from src.metrics import EventMetrics, PackMetrics

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def _extract_pack_id(path: Path) -> int:
    """ maps each graph file name to its PackId, which is used to link with labels.parquet and for final output """
    match = re.fullmatch(r'pack_(\d+)\.pt', path.name)
    if not match:
        raise click.ClickException(f"Nome file grafo non valido: {path.name}")
    return int(match.group(1))

def _load_snapshot(weights_path: Path, device: str) -> dict:
    """ Loads the model snapshot and checks for required keys. Returns a dictionary with 'state_dict', 'ip_dict', and optionally 'norm_stat_dict'. """
    snapshot = torch.load(weights_path.resolve(), map_location=device)
    if 'state_dict' not in snapshot or 'ip_dict' not in snapshot:
        raise click.ClickException("Snapshot non valido: chiavi richieste 'state_dict' e 'ip_dict' mancanti.")
    if 'norm_stat_dict' not in snapshot:
        snapshot['norm_stat_dict'] = None
    return snapshot


def calib_priors(train_prior: float, test_prior: float, gt_all: np.ndarray, scores_all: np.ndarray) -> np.ndarray:
    """ Applies Bayes prior-shift calibration to the scores_all array using the given train_prior and test_prior. Returns the calibrated scores. """

    if train_prior is None:
        raise click.ClickException("Snapshot mancante di 'train_prior' necessario per calibrazione prior-shift. ")
    if test_prior is None:
        test_pos = int((gt_all[:, 0] == 1).sum())
        test_neg = int((gt_all[:, 0] == 0).sum())
        test_prior = test_pos / (test_pos + test_neg) if (test_pos + test_neg) > 0 else 0.0

    click.echo(f"Calibrating priors: train_prior={train_prior:.6f}, test_prior={test_prior:.6f}")

    calibrated_scores, prior_ratio = bayesPriorShift(scores_all, train_prior, test_prior)
    click.echo(f"  Prior ratio: {prior_ratio:.6f}")
    click.echo(f"  Example: raw=0.99 -> calibrated={0.99 * prior_ratio / (0.99 * prior_ratio + 0.01):.6f}")
    return calibrated_scores, prior_ratio

@click.command()
@click.argument('inputdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.argument('outdir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.option('-w', '--weights', 'weights_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help='Path to the model snapshot (.pth).')
@click.option('-b', '--batch-size', 'batch_size', type=int, default=64, show_default=True, help='Batch size for test inference.')
@click.option('--threshold', type=float, default=0.5, show_default=True, help='Threshold for binary prediction from scores.')
@click.option('--cut', type=int, default=None, help='If set, cuts frames after the given number (as in training).')
@click.option('-e', '--event-metrics', is_flag=True, default=False, help='Compute event-level clustering metrics (FAR/h, event precision).')
@click.option('--sim-duration', type=int, default=60, show_default=True, help='Simulation duration in seconds (for FAR/h computation).')
@click.option('--calibrate-priors', is_flag=True, default=False, help='Apply Bayes prior-shift calibration. Uses snapshot train_prior.')
@click.option('--test-prior', type=float, default=None, help='Deployment P(y=1) for prior-shift calibration (computed from test data if not set).')
@click.option('--gap-pred', type=int, default=5, show_default=True, help='Gap (samples) for clustering prediction indices.')
@click.option('--gap-gt', type=int, default=20, show_default=True, help='Gap (samples) for clustering ground-truth indices.')
@click.option('--match-tol', type=int, default=20, show_default=True, help='Tolerance (samples) when matching predicted clusters to GT events.')
def main(inputdir: Path, outdir: Path, weights_path: Path, batch_size: int, threshold: float, cut: int | None, event_metrics: bool, sim_duration: int, calibrate_priors: bool, test_prior: float | None, gap_pred: int = 5, gap_gt: int = 20, match_tol: int = 20):
    ## ==================== CHECKS & SETUP ====================
    
    if not (0.0 <= threshold <= 1.0):
        raise click.ClickException("--threshold deve essere compreso tra 0.0 e 1.0")

    if not (inputdir / '.graphs').is_dir():
        raise click.ClickException(f"Directory di input {inputdir} non valida: sottodirectory '.graphs' mancante.")

    gpath = inputdir / '.graphs'
    metadata = MetaData.loadJson(gpath / 'metadata.json')
    active_labels = metadata.active_labels

    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ## ==================== MODEL & DATA ====================

    ### ---------- load model snapshot

    snapshot = _load_snapshot(weights_path, DEVICE)
    norm_stats = snapshot.get('norm_stat_dict')

    transform = []
    if cut is not None:
        transform.append(TFs.CutFrames(cut))
    transform = T.Compose(transform)

    if norm_stats is not None:
        zscore_mu_sigma = (norm_stats['mu'], norm_stats['sigma'])
        normalize_zscore = True
    else:
        zscore_mu_sigma = None
        normalize_zscore = False

    ### ---------- builds test dataset and dataloader;

    d_test = MapGraph(
        gpath,
        device=DEVICE,
        transform=transform,
        normalizeZScore=normalize_zscore,
        metadata=metadata,
        zscore_mu_sigma=zscore_mu_sigma,
    )
    if len(d_test) == 0:
        raise click.ClickException(f"Nessun grafo trovato in {gpath}")

    click.echo(f"Using device: {DEVICE}")
    click.echo(f"Test split dir: {inputdir}")
    click.echo(f"Test samples: {len(d_test)}")
    click.echo(f"Active labels: {active_labels}")

    model = GruSage(**snapshot['ip_dict']).to(DEVICE).eval()
    model.load_state_dict(snapshot['state_dict'])

    dl_test = GDL(d_test, batch_size=batch_size, shuffle=False)
    pack_ids = [_extract_pack_id(p) for p in d_test.paths]

    num_samples = len(d_test)
    num_labels = len(active_labels)
    scores_all = np.zeros((num_samples, num_labels), dtype=np.float32)
    preds_all = np.zeros((num_samples, num_labels), dtype=np.int32)
    gt_from_graphs = np.full((num_samples, num_labels), -1, dtype=np.int32)

    ### ---------- inference loop to collect scores, predictions, and GT from graphs (if available) into aligned arrays; raises exception if no scores are found in the CSV predictions

    cursor = 0
    with torch.inference_mode():
        for batch in dl_test:
            batch = batch.to(DEVICE)
            logits = model(batch)
            scores = torch.sigmoid(logits).detach().cpu().numpy().reshape(batch.num_graphs, num_labels)
            preds = (scores >= threshold).astype(np.int32)

            next_cursor = cursor + batch.num_graphs
            scores_all[cursor:next_cursor, :] = scores
            preds_all[cursor:next_cursor, :] = preds

            if hasattr(batch, 'y') and batch.y is not None:
                y = batch.y.float().view(batch.num_graphs, num_labels).detach().cpu().numpy().astype(np.int32)
                gt_from_graphs[cursor:next_cursor, :] = y

            cursor = next_cursor

    ## ==================== LOAD GROUND TRUTH LABELS ====================

    if not (gt_from_graphs >= 0).all():
        raise click.ClickException("Alcuni campioni non hanno etichette GT nei grafi. Assicurati che tutti i grafi contengano etichette GT valide per le active labels.")
    gt_all = gt_from_graphs

    if calibrate_priors:
        train_prior = snapshot.get('train_prior')
        scores_all, prior_ratio = calib_priors(train_prior, test_prior, gt_all, scores_all)

    preds_all = (scores_all >= threshold).astype(np.int32)


    ## ==================== METRICS & OUTPUT ====================

    for local_lb_idx, lb_value in enumerate(active_labels):
        gt = gt_all[:, local_lb_idx].astype(np.int32)
        scr = scores_all[:, local_lb_idx].astype(np.float32)

        pm = PackMetrics(
            gt_arr_1d=gt,
            scr_arr_1d=scr,
            threshold=threshold,
        )

        pm.printout()
        pm.plot_in_csv(outdir, lb_value)

        if event_metrics:
            metrics = EventMetrics(
                gt_arr_1d=gt,
                scr_arr_1d=scr,
                threshold=threshold,
                sim_duration_s=sim_duration,
                gap_pred=gap_pred,
                gap_gt=gap_gt,
                match_tol=match_tol,
            )
            metrics.printout()
            metrics.plot_in_csv(outdir, lb_value)
            metrics.plot_temporal_comparison(outdir / f"test_temporal_plot_lb{lb_value}.png")


if __name__ == '__main__':
    main()