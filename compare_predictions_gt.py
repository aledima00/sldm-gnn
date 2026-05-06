import csv
from bisect import bisect_right
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SAMPLE_MS = 10.0


def _cluster_by_time(times_s: np.ndarray, gap_s: float) -> list[np.ndarray]:
    if len(times_s) == 0:
        return []
    clusters = [[times_s[0]]]
    for i in range(1, len(times_s)):
        if times_s[i] - times_s[i - 1] <= gap_s:
            clusters[-1].append(times_s[i])
        else:
            clusters.append([times_s[i]])
    return [np.array(c) for c in clusters]


def _linear_interpolate(times_s, values, target_s):
    if not times_s or not values or len(times_s) != len(values):
        return None
    if len(times_s) == 1:
        return values[0] if abs(target_s - times_s[0]) < 1e-12 else None
    if target_s < times_s[0] or target_s > times_s[-1]:
        return None
    right = bisect_right(times_s, target_s)
    if right == 0:
        return values[0]
    if right >= len(times_s):
        return values[-1]
    left = right - 1
    t0, t1 = times_s[left], times_s[right]
    v0, v1 = values[left], values[right]
    if abs(t1 - t0) < 1e-12:
        return v0
    return v0 + (target_s - t0) / (t1 - t0) * (v1 - v0)


def _load_gt_events(gt_parquet: Path, active_label: int | None) -> tuple[list[int], list[float]]:
    trigger_times_s: list[float] = []
    gt_labels: list[int] = []

    gt_df = pd.read_parquet(gt_parquet)
    required = {"PackId", "MLBEncoded"}
    missing = required - set(gt_df.columns)
    if missing:
        raise click.ClickException(
            "Parquet ground truth invalido: colonne mancanti "
            + ", ".join(sorted(missing))
        )

    for row_idx, value in enumerate(gt_df["MLBEncoded"].tolist()):
        mlb = int(value)
        if active_label is not None:
            gt_bin = 1 if (mlb & (1 << active_label)) else 0
        else:
            gt_bin = 1 if mlb > 0 else 0
        gt_labels.append(gt_bin)
        if gt_bin == 1:
            trigger_times_s.append((row_idx * SAMPLE_MS) / 1000.0)

    return gt_labels, trigger_times_s


def _load_prediction_scores(pred_csv: Path) -> list[float]:
    scores: list[float] = []
    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise click.ClickException("CSV predictions senza header.")
        required = {"Scores"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise click.ClickException(
                "CSV predictions invalido: colonne mancanti "
                + ", ".join(sorted(missing))
            )
        for row in reader:
            value_raw = (row.get("Scores") or "").strip()
            if value_raw == ".":
                continue
            try:
                scores.append(float(value_raw))
            except ValueError:
                scores.append(np.nan)
    return scores


def _build_aligned_pairs(gt_labels, pred_scores, pred_times_s, stride, pack_size):
    pairs = []
    for gt_idx, gt_bin in enumerate(gt_labels):
        target_time_s = ((gt_idx + pack_size) * SAMPLE_MS) / 1000.0
        s = _linear_interpolate(pred_times_s, pred_scores, target_time_s)
        if s is not None and not np.isnan(s):
            pairs.append((gt_bin, s, target_time_s))
    return pairs


def _compute_event_metrics_online(gt_labels, pred_scores, pred_times_s, threshold,
                                   stride, pack_size, sim_duration_s, outdir, label_name):
    pairs = _build_aligned_pairs(gt_labels, pred_scores, pred_times_s, stride, pack_size)
    if not pairs:
        click.echo("  No aligned GT-prediction pairs found.")
        return

    gt_times = np.array([p[2] for p in pairs if p[0] == 1])
    gt_events = _cluster_by_time(gt_times, gap_s=2.0) if len(gt_times) > 0 else []

    pred_positive_times = np.array([p[2] for p in pairs if p[1] >= threshold])
    pred_clusters = _cluster_by_time(pred_positive_times, gap_s=1.0)

    detected = set()
    matched = set()
    for ci, pc in enumerate(pred_clusters):
        pc_s, pc_e = pc[0], pc[-1]
        for ei, ge in enumerate(gt_events):
            if pc_s <= ge[-1] + 2.0 and pc_e >= ge[0] - 2.0:
                detected.add(ei)
                matched.add(ci)

    false_alarms = len(pred_clusters) - len(matched)
    n_detected = len(detected)
    n_missed = len(gt_events) - n_detected
    far_per_hour = false_alarms / sim_duration_s * 3600 if sim_duration_s > 0 else 0
    prec_ev = n_detected / len(pred_clusters) if len(pred_clusters) > 0 else 0.0

    click.echo(f"\n  Event-Level Metrics ({label_name}, threshold={threshold:g}):")
    click.echo(f"    GT events: {len(gt_events)}")
    click.echo(f"    Prediction clusters: {len(pred_clusters)}")
    click.echo(f"    Detected: {n_detected}/{len(gt_events)}")
    click.echo(f"    Missed: {n_missed}")
    click.echo(f"    False alarm clusters: {false_alarms}")
    click.echo(f"    False alarm rate: {far_per_hour:.1f}/h")
    click.echo(f"    Event precision: {prec_ev:.3f}")

    pd.DataFrame([{
        'LabelName': label_name, 'Threshold': threshold,
        'SimDurationSeconds': sim_duration_s, 'GtEvents': len(gt_events),
        'PredClusters': len(pred_clusters), 'DetectedEvents': n_detected,
        'MissedEvents': n_missed, 'FalseAlarmClusters': false_alarms,
        'FalseAlarmRatePerHour': far_per_hour, 'EventPrecision': prec_ev,
    }]).to_csv(outdir / 'test_event_metrics.csv', index=False)

    return pred_clusters, matched


def _threshold_sweep_online(gt_labels, pred_scores, pred_times_s, stride, pack_size,
                             sim_duration_s, outdir, label_name):
    pairs = _build_aligned_pairs(gt_labels, pred_scores, pred_times_s, stride, pack_size)
    if not pairs:
        return

    gt_times = np.array([p[2] for p in pairs if p[0] == 1])
    gt_events = _cluster_by_time(gt_times, gap_s=2.0) if len(gt_times) > 0 else []

    click.echo(f"\n  Threshold Sweep ({label_name}, event-level):")
    click.echo(f"    {'Th':>6s}  {'Clusters':>8s}  {'Detected':>8s}  {'Missed':>6s}  "
                f"{'FAlarms':>7s}  {'FAR/h':>6s}  {'Prec/Ev':>8s}")
    click.echo("    " + "-" * 68)

    rows = []
    for th in [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99]:
        pos_times = np.array([p[2] for p in pairs if p[1] >= th])
        pred_clusters = _cluster_by_time(pos_times, gap_s=1.0) if len(pos_times) > 0 else []

        detected = set()
        matched = set()
        for ci, pc in enumerate(pred_clusters):
            pc_s, pc_e = pc[0], pc[-1]
            for ei, ge in enumerate(gt_events):
                if pc_s <= ge[-1] + 2.0 and pc_e >= ge[0] - 2.0:
                    detected.add(ei)
                    matched.add(ci)

        fa = len(pred_clusters) - len(matched)
        nd = len(detected)
        nm = len(gt_events) - nd
        far = fa / sim_duration_s * 3600 if sim_duration_s > 0 else 0
        prec = nd / len(pred_clusters) if len(pred_clusters) > 0 else 0.0

        click.echo(f"    {th:6.2f}  {len(pred_clusters):8d}  {nd:8d}  "
                    f"{nm:6d}  {fa:7d}  {far:6.1f}  {prec:8.3f}")
        rows.append({
            'LabelName': label_name, 'Threshold': th,
            'PredClusters': len(pred_clusters), 'DetectedEvents': nd,
            'MissedEvents': nm, 'FalseAlarmClusters': fa,
            'FARPerHour': far, 'EventPrecision': prec,
        })

    pd.DataFrame(rows).to_csv(outdir / f'test_threshold_sweep_{label_name.lower()}.csv', index=False)


def _plot_online_comparison(pred_times_s, pred_scores, gt_labels, gt_triggers_s, threshold,
                            stride, pack_size, outdir, pred_clusters, matched_clusters):
    gt_shift_s = (pack_size * SAMPLE_MS) / 1000.0
    gt_triggers_shifted_s = [t + gt_shift_s for t in gt_triggers_s]
    gt_total_time_s = ((max(len(gt_labels) - 1, 0)) * SAMPLE_MS) / 1000.0 + gt_shift_s
    pred_total_time_s = pred_times_s[-1] if pred_times_s else 0

    fig, (ax, ax_detail) = plt.subplots(2, 1, figsize=(16, 8),
                                         gridspec_kw={'height_ratios': [3, 1]})

    ax.plot(pred_times_s, pred_scores, color='#4a4abc', linewidth=1.2, alpha=0.85, label='Score')

    for t in gt_triggers_shifted_s:
        ax.axvline(x=t, color='red', alpha=0.4, linewidth=1.7, linestyle='-')

    ax.axhline(y=threshold, color='green', linewidth=1.5, linestyle='--', alpha=0.9,
                label=f'Threshold ({threshold:g})')

    if pred_clusters is not None:
        for ci, pc in enumerate(pred_clusters):
            color = '#22aa44' if matched_clusters is not None and ci in matched_clusters else '#dd6622'
            alpha = 0.18 if matched_clusters is not None and ci in matched_clusters else 0.10
            ax.axvspan(pc[0], pc[-1], alpha=alpha, color=color, linewidth=0)

    ax.set_title('Prediction Score vs Ground Truth Triggers', loc='left', fontsize=11, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim(bottom=-0.05, top=1.05)
    ax.set_xlim(left=0.0, right=max(gt_total_time_s, pred_total_time_s, 0.1))
    ax.grid(True, alpha=0.25)

    legend_elements = [
        plt.Line2D([0], [0], color='#4a4abc', linewidth=1.5, label='Score'),
        plt.Line2D([0], [0], color='red', linewidth=1.5, linestyle='-', label='GT trigger'),
        plt.Line2D([0], [0], color='green', linewidth=1.5, linestyle='--', label=f'Threshold ({threshold:g})'),
    ]
    if pred_clusters is not None:
        legend_elements += [
            plt.Rectangle((0, 0), 1, 1, color='#22aa44', alpha=0.25, label='Detected (TP)'),
            plt.Rectangle((0, 0), 1, 1, color='#dd6622', alpha=0.25, label='False alarm'),
        ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

    aligned_pairs = _build_aligned_pairs(
        gt_labels, pred_scores, pred_times_s, stride, pack_size
    )
    if aligned_pairs:
        pair_times = [p[2] for p in aligned_pairs]
        pair_preds = [1 if p[1] >= threshold else 0 for p in aligned_pairs]
        pair_gt = [p[0] for p in aligned_pairs]
        colors = ['#22aa44' if pair_gt[i] else '#dd6622' for i in range(len(pair_preds))]
        ax_detail.bar(pair_times, [1.0 if p == 1 else 0.05 for p in pair_preds],
                       width=(SAMPLE_MS * stride) / 1000.0 * 0.8, color=colors, linewidth=0, alpha=0.7)
        ax_detail.set_ylabel('Prediction')
        ax_detail.set_yticks([0, 1])
        ax_detail.set_yticklabels(['0', '1'])
        ax_detail.set_ylim(bottom=-0.1, top=1.1)

    ax_detail.set_xlabel('Time [s]')
    ax_detail.set_xlim(left=0.0, right=ax.get_xlim()[1])
    ax_detail.grid(True, alpha=0.15, axis='y')

    n_detected = len(matched_clusters) if matched_clusters is not None else 0
    n_fa = (len(pred_clusters) - n_detected) if pred_clusters is not None else 0
    info_text = (
        f"GT triggers: {len(gt_triggers_s)} | Predictions: {len(pred_scores)} | "
        f"Stride: {stride} | Pack size: {pack_size} ({gt_shift_s * 1000:g}ms shift) | "
        f"Threshold: {threshold:g}"
    )
    if pred_clusters is not None:
        info_text += (
            f"\nPrediction clusters: {len(pred_clusters)} | "
            f"Detected: {n_detected}/{len(gt_triggers_s)} | "
            f"False alarm clusters: {n_fa}"
        )
    ax.text(0.99, 1.07, info_text, transform=ax.transAxes, fontsize=9,
            color='#444444', va='bottom', ha='right')

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    plot_path = outdir / 'compare_predictions_gt.png'
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    click.echo(f"  Plot saved to {plot_path}")


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--gt-parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help='Parquet ground truth con colonne ["PackId","MLBEncoded"].')
@click.option("--pred-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True, help="CSV predictions (output of rcv.py) con colonne [PredictionLabels,Scores].")
@click.option("-S", "--stride", type=click.IntRange(min=1), required=True,
              help="Stride usato in inferenza: un sample predetto ogni <stride> sample GT.")
@click.option("--pack-size", type=click.IntRange(min=0), default=0, show_default=True,
              help="Numero di frame (da 10ms) usati per prediction; shift in avanti delle GT.")
@click.option("--threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.5, show_default=True,
              help="Soglia per classificare i prediction scores in positivi/negativi.")
@click.option("--outdir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              default=None, help="Directory di output per metriche, plot e CSV.")
@click.option("-e", "--event-metrics", is_flag=True, default=False,
              help="Compute event-level clustering metrics (FAR/h, event precision).")
@click.option("--threshold-sweep", is_flag=True, default=False,
              help="Sweep thresholds and report event-level trade-off table + CSV.")
@click.option("--sim-duration", type=int, default=600, show_default=True,
              help="Simulation duration in seconds (for FAR/h computation).")
@click.option("--active-label", type=int, default=None,
              help="Active label index to extract from MLBEncoded bitmask (e.g. 2 for TURN).")
def main(gt_parquet, pred_csv, stride, pack_size, threshold, outdir, event_metrics,
         threshold_sweep, sim_duration, active_label):
    """Confronta temporalmente GT e prediction scores con metriche event-level."""
    if outdir is None:
        outdir = Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    gt_labels, gt_triggers_s = _load_gt_events(gt_parquet, active_label)
    pred_scores = _load_prediction_scores(pred_csv)

    if not pred_scores:
        raise click.ClickException("Nessuno score trovato nel CSV predictions.")

    pred_times_s = [((i * stride) * SAMPLE_MS) / 1000.0 for i in range(len(pred_scores))]

    pairs = _build_aligned_pairs(gt_labels, pred_scores, pred_times_s, stride, pack_size)
    if not pairs:
        raise click.ClickException("Nessuna coppia GT-prediction allineata temporalmente.")

    gt_arr = np.array([p[0] for p in pairs], dtype=np.int32)
    scr_arr = np.array([p[1] for p in pairs], dtype=np.float32)
    pred_arr = (scr_arr >= threshold).astype(np.int32)

    n_gt_pos = int((gt_arr == 1).sum())
    n_gt_neg = int((gt_arr == 0).sum())
    click.echo(f"Aligned pairs: {len(pairs)} (GT pos={n_gt_pos}, GT neg={n_gt_neg})")

    tp = int(((pred_arr == 1) & (gt_arr == 1)).sum())
    tn = int(((pred_arr == 0) & (gt_arr == 0)).sum())
    fp = int(((pred_arr == 1) & (gt_arr == 0)).sum())
    fn = int(((pred_arr == 0) & (gt_arr == 1)).sum())
    acc = float((pred_arr == gt_arr).mean())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    roc = float(roc_auc_score(gt_arr, scr_arr)) if len(np.unique(gt_arr)) > 1 else np.nan
    ap = float(average_precision_score(gt_arr, scr_arr)) if len(np.unique(gt_arr)) > 1 else np.nan

    click.echo(f"\n--- Results ---")
    click.echo(f"  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  "
                f"F1={f1:.4f}  ROC-AUC={roc:.4f}  AvgPrecision={ap:.4f}")
    click.echo(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")

    label_name = f"label_{active_label}" if active_label is not None else "all"
    pd.DataFrame([{
        'LabelName': label_name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
        'F1': f1, 'ROCAUC': roc, 'AveragePrecision': ap,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn, 'NumAlignedPairs': len(pairs),
        'Threshold': threshold, 'Stride': stride, 'PackSize': pack_size,
        'GtPositives': n_gt_pos, 'GtNegatives': n_gt_neg,
    }]).to_csv(outdir / 'test_metrics_summary.csv', index=False)

    pd.DataFrame([{
        'PairIndex': i, 'GtBin': int(p[0]), 'Score': float(p[1]),
        'TimeS': float(p[2]), 'Prediction': int(scr_arr[i] >= threshold),
    } for i, p in enumerate(pairs)]).to_csv(outdir / 'test_predictions_vs_gt.csv', index=False)

    pred_clusters = None
    matched_clusters = None
    if event_metrics:
        result = _compute_event_metrics_online(
            gt_labels, pred_scores, pred_times_s, threshold,
            stride, pack_size, sim_duration, outdir, label_name
        )
        if result:
            pred_clusters, matched_clusters = result

    if threshold_sweep:
        _threshold_sweep_online(
            gt_labels, pred_scores, pred_times_s, stride, pack_size,
            sim_duration, outdir, label_name
        )

    _plot_online_comparison(
        pred_times_s, pred_scores, gt_labels, gt_triggers_s, threshold,
        stride, pack_size, outdir, pred_clusters, matched_clusters
    )


if __name__ == "__main__":
    main()
