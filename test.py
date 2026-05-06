import json
import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from torch_geometric.loader import DataLoader as GDL

import src.transforms as TFs
from src.dataset import MapGraph
from src.models.grusage import GruSage
from src.utils import MetaData, getLbName


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _cluster(idx_array: np.ndarray, gap: int) -> list[np.ndarray]:
    if len(idx_array) == 0:
        return []
    clusters = [[idx_array[0]]]
    for i in range(1, len(idx_array)):
        if idx_array[i] - idx_array[i - 1] <= gap:
            clusters[-1].append(idx_array[i])
        else:
            clusters.append([idx_array[i]])
    return [np.array(c) for c in clusters]


def _compute_event_metrics(gt: np.ndarray, scores: np.ndarray, threshold: float,
                           sim_duration_s: int, outpath: Path, label_name: str):
    preds = (scores >= threshold).astype(np.int32)

    gt_idx = np.where(gt == 1)[0]
    gt_events = _cluster(gt_idx, gap=20)
    if not gt_events:
        click.echo("  No GT events found, skipping event-level metrics.")
        return

    pred_idx = np.where(preds == 1)[0]
    pred_clusters = _cluster(pred_idx, gap=5)

    detected_events = set()
    matched_clusters = set()
    for ci, pc in enumerate(pred_clusters):
        pc_start, pc_end = pc[0], pc[-1]
        for ei, ge in enumerate(gt_events):
            gs, ge_end = ge[0], ge[-1]
            if pc_start <= ge_end + 20 and pc_end >= gs - 20:
                detected_events.add(ei)
                matched_clusters.add(ci)

    false_alarms = len(pred_clusters) - len(matched_clusters)
    n_detected = len(detected_events)
    n_missed = len(gt_events) - n_detected
    far_per_hour = false_alarms / sim_duration_s * 3600
    event_precision = n_detected / len(pred_clusters) if len(pred_clusters) > 0 else 0.0

    click.echo(f"\n  Event-Level Metrics ({label_name}, threshold={threshold:g}):")
    click.echo(f"    GT events: {len(gt_events)}")
    click.echo(f"    Prediction clusters: {len(pred_clusters)}")
    click.echo(f"    Detected: {n_detected}/{len(gt_events)}")
    click.echo(f"    Missed: {n_missed}")
    click.echo(f"    False alarm clusters: {false_alarms}")
    click.echo(f"    False alarm rate: {far_per_hour:.1f}/h")
    click.echo(f"    Event precision: {event_precision:.3f}")

    event_rows = [{
        'LabelName': label_name,
        'Threshold': threshold,
        'SimDurationSeconds': sim_duration_s,
        'GtEvents': len(gt_events),
        'PredClusters': len(pred_clusters),
        'DetectedEvents': n_detected,
        'MissedEvents': n_missed,
        'FalseAlarmClusters': false_alarms,
        'FalseAlarmRatePerHour': far_per_hour,
        'EventPrecision': event_precision,
    }]
    pd.DataFrame(event_rows).to_csv(outpath / 'test_event_metrics.csv', index=False)
    click.echo(f"    Saved to {outpath / 'test_event_metrics.csv'}")


def _threshold_sweep_report(gt: np.ndarray, scores: np.ndarray, active_labels: list[int],
                            outpath: Path, sim_duration_s: int):
    click.echo("\n  Threshold Sweep (per-label, event-level):")
    for local_lb_idx, global_lb in enumerate(active_labels):
        g = gt[:, local_lb_idx].astype(np.int32)
        s = scores[:, local_lb_idx].astype(np.float32)
        lb_name = getLbName(local_lb_idx, active_labels)

        gt_idx = np.where(g == 1)[0]
        gt_events = _cluster(gt_idx, gap=20)

        rows = []
        click.echo(f"\n  {lb_name}:")
        click.echo(f"    {'Th':>6s}  {'Clusters':>8s}  {'Detected':>8s}  {'Missed':>6s}  "
                    f"{'FAlarms':>7s}  {'FAR/h':>6s}  {'Prec/Ev':>8s}")
        click.echo("    " + "-" * 68)

        for th in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
            preds = (s >= th).astype(np.int32)
            pred_idx = np.where(preds == 1)[0]
            pred_clusters = _cluster(pred_idx, gap=5)

            detected = set()
            matched = set()
            for ci, pc in enumerate(pred_clusters):
                pc_start, pc_end = pc[0], pc[-1]
                for ei, ge in enumerate(gt_events):
                    gs, ge_end = ge[0], ge[-1]
                    if pc_start <= ge_end + 20 and pc_end >= gs - 20:
                        detected.add(ei)
                        matched.add(ci)

            false_alarms = len(pred_clusters) - len(matched)
            n_detected = len(detected)
            n_missed = len(gt_events) - n_detected
            far = false_alarms / sim_duration_s * 3600
            prec = n_detected / len(pred_clusters) if len(pred_clusters) > 0 else 0.0

            click.echo(f"    {th:6.2f}  {len(pred_clusters):8d}  {n_detected:8d}  "
                        f"{n_missed:6d}  {false_alarms:7d}  {far:6.1f}  {prec:8.3f}")

            rows.append({
                'LabelName': lb_name, 'Threshold': th,
                'PredClusters': len(pred_clusters), 'DetectedEvents': n_detected,
                'MissedEvents': n_missed, 'FalseAlarmClusters': false_alarms,
                'FARPerHour': far, 'EventPrecision': prec,
            })

        pd.DataFrame(rows).to_csv(outpath / f'test_threshold_sweep_{lb_name.lower()}.csv', index=False)


def _apply_prior_shift(scores: np.ndarray, train_pos_frac: float,
                       test_pos_frac: float) -> np.ndarray:
    train_neg_frac = 1.0 - train_pos_frac
    test_neg_frac = 1.0 - test_pos_frac
    prior_ratio = (test_pos_frac / test_neg_frac) / (train_pos_frac / train_neg_frac)
    calibrated = scores * prior_ratio / (scores * prior_ratio + (1.0 - scores))
    return calibrated, prior_ratio


def _resolve_test_split_dir(inputdir: Path) -> Path:
    inputdir = inputdir.resolve()
    candidate_root = inputdir / 'test'
    if (candidate_root / '.graphs').is_dir():
        return candidate_root
    if (inputdir / '.graphs').is_dir():
        return inputdir
    raise click.ClickException(
        f"Impossibile trovare lo split test: atteso '{inputdir / 'test' / '.graphs'}' oppure '{inputdir / '.graphs'}'."
    )


def _extract_pack_id(path: Path) -> int:
    match = re.fullmatch(r'pack_(\d+)\.pt', path.name)
    if not match:
        raise click.ClickException(f"Nome file grafo non valido: {path.name}")
    return int(match.group(1))


def _decode_mlb(mlb_encoded: int, active_labels: list[int]) -> list[int]:
    out = [0] * len(active_labels)
    for i, lb in enumerate(active_labels):
        out[i] = 1 if (int(mlb_encoded) & (1 << lb)) else 0
    return out


def _plot_temporal_comparison(scores_all: np.ndarray, preds_all: np.ndarray, gt_all: np.ndarray,
                              active_labels: list[int], threshold: float, outpath: Path,
                              event_clusters: list[np.ndarray] | None = None,
                              matched_indices: set | None = None):
    num_labels = len(active_labels)
    if num_labels == 1:
        fig, (ax, ax_detail) = plt.subplots(2, 1, figsize=(16, 8),
                                             gridspec_kw={'height_ratios': [3, 1]})
        local_lb_idx = 0
        scr = scores_all[:, local_lb_idx]
        gt = gt_all[:, local_lb_idx]
        pred = preds_all[:, local_lb_idx]

        x_axis = np.arange(len(scr))
        ax.plot(x_axis, scr, color='#4a4abc', linewidth=1.2, alpha=0.85, label='Score')

        for idx in np.where(gt == 1)[0]:
            ax.axvline(x=idx, color='red', alpha=0.4, linewidth=1.7, linestyle='-')

        ax.axhline(y=threshold, color='green', linewidth=1.5, linestyle='--', alpha=0.9,
                    label=f'Threshold ({threshold:g})')

        if event_clusters is not None and matched_indices is not None:
            for ci, pc in enumerate(event_clusters):
                color = '#22aa44' if ci in matched_indices else '#dd6622'
                alpha = 0.18 if ci in matched_indices else 0.10
                ax.axvspan(pc[0], pc[-1], alpha=alpha, color=color, linewidth=0)

        lb_name = getLbName(local_lb_idx, active_labels)
        ax.set_title(f'Score vs Ground Truth Events — {lb_name}', loc='left',
                      fontsize=11, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(bottom=-0.05, top=1.05)
        ax.grid(True, alpha=0.25)

        legend_elements = [
            plt.Line2D([0], [0], color='#4a4abc', linewidth=1.5, label='Score'),
            plt.Line2D([0], [0], color='red', linewidth=1.5, linestyle='-', label='GT event'),
            plt.Line2D([0], [0], color='green', linewidth=1.5, linestyle='--', label=f'Threshold ({threshold:g})'),
        ]
        if event_clusters is not None:
            legend_elements += [
                plt.Rectangle((0, 0), 1, 1, color='#22aa44', alpha=0.25, label='Detected (TP)'),
                plt.Rectangle((0, 0), 1, 1, color='#dd6622', alpha=0.25, label='False alarm'),
            ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        num_events = int((gt == 1).sum())
        num_pred_positive = int((pred == 1).sum())
        num_samples = len(scr)

        n_detected = len(matched_indices) if matched_indices is not None else 0
        n_fa = (len(event_clusters) - n_detected) if event_clusters is not None else 0
        info_text = (
            f"Samples: {num_samples} | GT events: {num_events} | "
            f"Pred +: {num_pred_positive} | Threshold: {threshold:g}"
        )
        if event_clusters is not None:
            info_text += (
                f"\nPrediction clusters: {len(event_clusters)} | "
                f"Detected: {n_detected}/{num_events} | "
                f"False alarm clusters: {n_fa}"
            )
        ax.text(0.99, 1.07, info_text, transform=ax.transAxes, fontsize=9,
                color='#444444', va='bottom', ha='right')

        ax_detail.bar(x_axis, pred, color=['#22aa44' if gt[i] else '#dd6622' for i in range(len(pred))],
                      width=1.0, linewidth=0)
        ax_detail.set_xlabel('Sample Index')
        ax_detail.set_ylabel('Prediction')
        ax_detail.set_yticks([0, 1])
        ax_detail.set_yticklabels(['0', '1'])
        ax_detail.set_ylim(bottom=-0.1, top=1.1)
        ax_detail.grid(True, alpha=0.15, axis='y')

        plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        plot_path = outpath / f'test_temporal_plot_{getLbName(local_lb_idx, active_labels).lower()}.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
    else:
        fig, axes = plt.subplots(num_labels, 1, figsize=(14, 4 * num_labels))
        if num_labels == 1:
            axes = [axes]

        for local_lb_idx, ax in enumerate(axes):
            scr = scores_all[:, local_lb_idx]
            gt = gt_all[:, local_lb_idx]

            x_axis = np.arange(len(scr))
            ax.plot(x_axis, scr, color='#4a4abc', linewidth=1.7, marker='o',
                     markersize=3.0, markeredgewidth=0.0, label='Score')

            for idx in np.where(gt == 1)[0]:
                ax.axvline(x=idx, color='red', alpha=0.5, linewidth=1.7, linestyle='-')

            ax.axhline(y=threshold, color='green', linewidth=1.5, linestyle='--', alpha=0.9,
                        label=f'Threshold ({threshold:g})')

            lb_name = getLbName(local_lb_idx, active_labels)
            ax.set_title(f'Score vs Ground Truth Events — {lb_name}', loc='left')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Score')
            ax.set_ylim(bottom=-0.05, top=1.05)
            ax.grid(True, alpha=0.25)
            ax.legend(loc='upper right')

        plt.tight_layout()
        plot_path = outpath / 'test_temporal_plots.png'
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)


def _load_snapshot(weights_path: Path, device: str) -> dict:
    snapshot = torch.load(weights_path.resolve(), map_location=device)
    if 'state_dict' not in snapshot or 'ip_dict' not in snapshot:
        raise click.ClickException("Snapshot non valido: chiavi richieste 'state_dict' e 'ip_dict' mancanti.")
    if 'norm_stat_dict' not in snapshot:
        snapshot['norm_stat_dict'] = None
    return snapshot


def _load_gt_from_labels_parquet(labels_path: Path, pack_ids: list[int], active_labels: list[int]) -> np.ndarray:
    if not labels_path.exists():
        raise click.ClickException(
            "GT non disponibile nei grafi e file labels.parquet assente: impossibile confrontare prediction e ground truth."
        )

    labels_df = pd.read_parquet(labels_path)
    required_cols = {'PackId', 'MLBEncoded'}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise click.ClickException(
            f"labels.parquet invalido: colonne mancanti {sorted(missing)}"
        )

    labels_df = labels_df[['PackId', 'MLBEncoded']].drop_duplicates(subset=['PackId'])
    pid_to_mlb = dict(zip(labels_df['PackId'].astype(int).tolist(), labels_df['MLBEncoded'].astype(int).tolist()))

    gt = np.full((len(pack_ids), len(active_labels)), -1, dtype=np.int32)
    missing_pack_ids: list[int] = []
    for i, pid in enumerate(pack_ids):
        mlb = pid_to_mlb.get(pid)
        if mlb is None:
            missing_pack_ids.append(pid)
            continue
        gt[i, :] = np.array(_decode_mlb(mlb, active_labels), dtype=np.int32)

    if missing_pack_ids:
        sample = missing_pack_ids[:10]
        raise click.ClickException(
            f"Ground truth mancante per {len(missing_pack_ids)} PackId (esempi: {sample})."
        )

    return gt


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('inputdir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.argument('outdir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path), required=True, nargs=1)
@click.option('-w', '--weights', 'weights_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help='Path to the model snapshot (.pth).')
@click.option('-b', '--batch-size', 'batch_size', type=int, default=64, show_default=True, help='Batch size for test inference.')
@click.option('--threshold', type=float, default=0.5, show_default=True, help='Threshold for binary prediction from scores.')
@click.option('--cut', type=int, default=None, help='If set, cuts frames after the given number (as in training).')
@click.option('-e', '--event-metrics', is_flag=True, default=False, help='Compute event-level clustering metrics (FAR/h, event precision).')
@click.option('--threshold-sweep', is_flag=True, default=False, help='Sweep thresholds and report event-level trade-off table + CSV.')
@click.option('--sim-duration', type=int, default=600, show_default=True, help='Simulation duration in seconds (for FAR/h computation).')
@click.option('--calibrate-priors', is_flag=True, default=False, help='Apply Bayes prior-shift calibration using --train-metadata and test metadata.')
@click.option('--train-metadata', 'train_metadata_path', type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default=None, help='Path to train/.graphs/metadata.json for prior-shift calibration.')
def main(inputdir: Path, outdir: Path, weights_path: Path, batch_size: int, threshold: float,
         cut: int | None, event_metrics: bool, threshold_sweep: bool, sim_duration: int,
         calibrate_priors: bool, train_metadata_path: Path | None):
    if not (0.0 <= threshold <= 1.0):
        raise click.ClickException("--threshold deve essere compreso tra 0.0 e 1.0")

    split_dir = _resolve_test_split_dir(inputdir)
    gpath = split_dir / '.graphs'
    metadata = MetaData.loadJson(gpath / 'metadata.json')
    active_labels = metadata.active_labels

    outpath = outdir.resolve()
    outpath.mkdir(parents=True, exist_ok=True)

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
    click.echo(f"Test split dir: {split_dir}")
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

    if (gt_from_graphs >= 0).all():
        gt_all = gt_from_graphs
    else:
        gt_all = _load_gt_from_labels_parquet(split_dir / 'labels.parquet', pack_ids, active_labels)

    if calibrate_priors:
        if train_metadata_path is None:
            raise click.ClickException(
                "--calibrate-priors requires --train-metadata PATH to train/.graphs/metadata.json"
            )
        train_meta = MetaData.loadJson(train_metadata_path)

        test_pos = int((gt_all[:, 0] == 1).sum())
        test_neg = int((gt_all[:, 0] == 0).sum())
        train_pos_frac = train_meta.n_positive / train_meta.n_samples
        test_pos_frac = test_pos / (test_pos + test_neg) if (test_pos + test_neg) > 0 else 0.0

        click.echo(f"Prior-shift calibration:")
        click.echo(f"  Train P(y=1)={train_pos_frac:.4f}, Test P(y=1)={test_pos_frac:.6f}")
        scores_calibrated, prior_ratio = _apply_prior_shift(scores_all, train_pos_frac, test_pos_frac)
        click.echo(f"  Prior ratio: {prior_ratio:.6f}")
        click.echo(f"  Example: raw=0.99 -> calibrated={0.99 * prior_ratio / (0.99 * prior_ratio + 0.01):.6f}")
        scores_all = scores_calibrated

    preds_all = (scores_all >= threshold).astype(np.int32)

    metrics_rows: list[dict] = []
    details_rows: list[dict] = []

    for sample_idx, pid in enumerate(pack_ids):
        for local_lb_idx, global_lb in enumerate(active_labels):
            details_rows.append({
                'PackId': pid,
                'ActiveLabelIndex': local_lb_idx,
                'GlobalLabelId': global_lb,
                'LabelName': getLbName(local_lb_idx, active_labels),
                'GroundTruth': int(gt_all[sample_idx, local_lb_idx]),
                'Prediction': int(preds_all[sample_idx, local_lb_idx]),
                'Score': float(scores_all[sample_idx, local_lb_idx]),
            })

    for local_lb_idx, global_lb in enumerate(active_labels):
        gt = gt_all[:, local_lb_idx].astype(np.int32)
        pred = preds_all[:, local_lb_idx].astype(np.int32)
        scr = scores_all[:, local_lb_idx].astype(np.float32)

        cm = confusion_matrix(gt, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt,
            pred,
            average='binary',
            zero_division=0,
        )
        accuracy = float((pred == gt).mean())
        roc_auc = float(roc_auc_score(gt, scr)) if np.unique(gt).size > 1 else np.nan
        ap = float(average_precision_score(gt, scr)) if np.unique(gt).size > 1 else np.nan

        lb_name = getLbName(local_lb_idx, active_labels)
        metrics_rows.append({
            'ActiveLabelIndex': local_lb_idx,
            'GlobalLabelId': global_lb,
            'LabelName': lb_name,
            'Accuracy': accuracy,
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'ROCAUC': roc_auc,
            'AveragePrecision': ap,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'NumSamples': int(gt.shape[0]),
        })

        click.echo(f"\n--- {lb_name} (label {global_lb}) ---")
        click.echo(f"  Accuracy={accuracy:.4f}  Precision={precision:.4f}  Recall={recall:.4f}  "
                    f"F1={f1:.4f}  ROC-AUC={roc_auc:.4f}  AvgPrecision={ap:.4f}")
        click.echo(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

        if event_metrics:
            _compute_event_metrics(gt, scr, threshold, sim_duration, outpath, lb_name)

    details_df = pd.DataFrame(details_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    details_path = outpath / 'test_predictions_vs_gt.csv'
    metrics_path = outpath / 'test_metrics_summary.csv'
    details_df.to_csv(details_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    plot_clusters = None
    plot_matched = None
    if event_metrics and num_labels == 1:
        gt_evt = gt_all[:, 0].astype(np.int32)
        prd_evt = (scores_all[:, 0] >= threshold).astype(np.int32)
        gt_idx_evt = np.where(gt_evt == 1)[0]
        gt_events_evt = _cluster(gt_idx_evt, gap=20)
        prd_idx_evt = np.where(prd_evt == 1)[0]
        pred_clusters_evt = _cluster(prd_idx_evt, gap=5)
        matched_evt = set()
        for ci, pc in enumerate(pred_clusters_evt):
            pc_start, pc_end = pc[0], pc[-1]
            for ei, ge in enumerate(gt_events_evt):
                gs, ge_end = ge[0], ge[-1]
                if pc_start <= ge_end + 20 and pc_end >= gs - 20:
                    matched_evt.add(ci)
        plot_clusters = pred_clusters_evt
        plot_matched = matched_evt

    _plot_temporal_comparison(scores_all, preds_all, gt_all, active_labels, threshold, outpath,
                              event_clusters=plot_clusters, matched_indices=plot_matched)

    if threshold_sweep:
        _threshold_sweep_report(gt_all, scores_all, active_labels, outpath, sim_duration)

    overall_accuracy = float((preds_all == gt_all).mean())
    click.echo(f"\nOverall multilabel accuracy: {overall_accuracy:.4f}")
    click.echo(f"Saved detailed predictions vs gt: {details_path}")
    click.echo(f"Saved metrics summary: {metrics_path}")
    click.echo(f"Saved temporal plot(s) to {outpath}")


if __name__ == '__main__':
    main()