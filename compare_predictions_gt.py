import csv
from bisect import bisect_right
from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd


SAMPLE_MS = 10.0


def _moving_average(values: list[float], window_size: int) -> list[float]:
    """Compute trailing moving average with fixed window size in samples."""
    if window_size <= 1:
        return values.copy()

    out: list[float] = []
    running_sum = 0.0

    for i, v in enumerate(values):
        running_sum += v
        if i >= window_size:
            running_sum -= values[i - window_size]

        current_count = min(i + 1, window_size)
        out.append(running_sum / current_count)

    return out


def _linear_interpolate(times_s: list[float], values: list[float], target_s: float) -> float | None:
    """Linearly interpolate value at target time over a monotonic timeline."""
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
    t0 = times_s[left]
    t1 = times_s[right]
    v0 = values[left]
    v1 = values[right]

    if abs(t1 - t0) < 1e-12:
        return v0

    alpha = (target_s - t0) / (t1 - t0)
    return v0 + alpha * (v1 - v0)


def _load_gt_events(gt_parquet: Path) -> tuple[list[int], list[float]]:
    """Load GT labels and return (binary_labels, trigger_times_s)."""
    trigger_times_s: list[float] = []
    gt_labels: list[int] = []

    try:
        gt_df = pd.read_parquet(gt_parquet)
    except Exception as exc:
        raise click.ClickException(f"Impossibile leggere il Parquet ground truth: {exc}") from exc

    required = {"PackId", "MLBEncoded"}
    missing = required - set(gt_df.columns)
    if missing:
        raise click.ClickException(
            "Parquet ground truth invalido: colonne mancanti "
            + ", ".join(sorted(missing))
        )

    for row_idx, value_raw in enumerate(gt_df["MLBEncoded"].tolist()):
        try:
            label_value = float(value_raw)
        except (TypeError, ValueError) as exc:
            raise click.ClickException(
                f"Valore MLBEncoded non numerico alla riga {row_idx + 1}: {value_raw!r}"
            ) from exc

        if pd.isna(label_value):
            raise click.ClickException(
                f"Valore MLBEncoded non valido alla riga {row_idx + 1}: {value_raw!r}"
            )

        gt_bin = 1 if label_value >= 0.5 else 0
        gt_labels.append(gt_bin)

        if gt_bin == 1:
            time_s = (row_idx * SAMPLE_MS) / 1000.0
            trigger_times_s.append(time_s)

    return gt_labels, trigger_times_s


def _load_prediction_scores(pred_csv: Path) -> list[float]:
    """Load prediction scores from CSV."""
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

        for row_idx, row in enumerate(reader):
            value_raw = (row.get("Scores") or "").strip()

            try:
                score_value = float(value_raw)
            except ValueError as exc:
                raise click.ClickException(
                    f"Valore Scores non numerico alla riga {row_idx + 2}: {value_raw!r}"
                ) from exc

            scores.append(score_value)

    return scores


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--gt-parquet",
    "gt_parquet",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help='Parquet ground truth con colonne ["PackId","MLBEncoded"].',
)
@click.option(
    "--pred-csv",
    "pred_csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="CSV predictions con colonne [PredictionLabels,Scores].",
)
@click.option(
    "--stride",
    type=click.IntRange(min=1),
    required=True,
    help="Stride usato in inferenza: un sample predetto ogni <stride> sample GT.",
)
@click.option(
    "--out",
    "out_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path opzionale per salvare la figura (es. compare.png).",
)
@click.option(
    "--show/--no-show",
    default=True,
    help="Mostra la figura a video.",
)
@click.option(
    "--smooth-window-ms",
    type=click.FloatRange(min=0.1),
    default=None,
    help="Finestra moving average in millisecondi per lo score smussato (opzionale).",
)
@click.option(
    "--pack-size",
    type=click.IntRange(min=0),
    default=0,
    show_default=True,
    help="Numero di frame (da 10ms) usati per prediction; shift in avanti delle GT.",
)
@click.option(
    "--threshold",
    type=click.FloatRange(min=0.0, max=1.0),
    default=0.5,
    show_default=True,
    help="Soglia per classificare i prediction scores in positivi/negativi.",
)
def main(
    gt_parquet: Path,
    pred_csv: Path,
    stride: int,
    out_path: Path | None,
    show: bool,
    smooth_window_ms: float | None,
    pack_size: int,
    threshold: float,
) -> None:
    """Confronta temporalmente GT e prediction scores con densita diverse."""
    gt_labels, gt_triggers_s = _load_gt_events(gt_parquet)
    gt_count = len(gt_labels)
    pred_scores = _load_prediction_scores(pred_csv)

    if not pred_scores:
        raise click.ClickException("Nessuno score trovato nel CSV predictions.")

    gt_shift_s = (pack_size * SAMPLE_MS) / 1000.0
    gt_triggers_shifted_s = [t + gt_shift_s for t in gt_triggers_s]

    pred_times_s = [((i * stride) * SAMPLE_MS) / 1000.0 for i in range(len(pred_scores))]
    pred_step_ms = SAMPLE_MS * stride
    smooth_window_samples: int | None = None
    pred_scores_smooth: list[float] | None = None

    if smooth_window_ms is not None:
        smooth_window_samples = max(1, int(round(smooth_window_ms / pred_step_ms)))
        pred_scores_smooth = _moving_average(pred_scores, smooth_window_samples)

    scores_for_classification = pred_scores_smooth if pred_scores_smooth is not None else pred_scores

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    pred_positive = 0
    pred_negative = 0
    valid_pairs = 0
    for gt_idx, gt_bin in enumerate(gt_labels):
        target_time_s = ((gt_idx + pack_size) * SAMPLE_MS) / 1000.0
        interp_score = _linear_interpolate(pred_times_s, scores_for_classification, target_time_s)
        if interp_score is None:
            continue

        pred_bin = 1 if interp_score >= threshold else 0
        valid_pairs += 1
        if pred_bin == 1:
            pred_positive += 1
        else:
            pred_negative += 1

        if pred_bin == 1 and gt_bin == 1:
            tp += 1
        elif pred_bin == 1 and gt_bin == 0:
            fp += 1
        elif pred_bin == 0 and gt_bin == 1:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    precision_txt = f"{precision:.4f}" if precision is not None else "N/A"
    recall_txt = f"{recall:.4f}" if recall is not None else "N/A"

    gt_total_time_s = ((max(gt_count - 1, 0)) * SAMPLE_MS) / 1000.0 + gt_shift_s
    pred_total_time_s = pred_times_s[-1]

    fig, ax = plt.subplots(figsize=(14, 5))
    raw_alpha = 0.5 if pred_scores_smooth is not None else 1.0

    ax.plot(
        pred_times_s,
        pred_scores,
        color="#ff7f0e",
        linewidth=1.7,
        alpha=raw_alpha,
        marker="o",
        markersize=3.0,
        markeredgewidth=0.0,
        label="Prediction score",
    )

    if pred_scores_smooth is not None:
        ax.plot(
            pred_times_s,
            pred_scores_smooth,
            color="#1f77b4",
            linewidth=2.0,
            alpha=0.7,
            label="Smoothed prediction score",
        )

    first_trigger = True
    for t in gt_triggers_shifted_s:
        ax.axvline(
            x=t,
            color="red",
            alpha=0.5,
            linewidth=1.7,
            linestyle="-",
            label="GT trigger" if first_trigger else None,
        )
        first_trigger = False

    ax.axhline(
        y=threshold,
        color="green",
        linewidth=1.5,
        linestyle="--",
        alpha=0.9,
        label=f"Threshold ({threshold:g})",
    )

    ax.set_title("Ground Truth Trigger vs Prediction Score", loc="left")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Score")
    ax.set_xlim(left=0.0, right=max(gt_total_time_s, pred_total_time_s, 0.1))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    info_line_1 = (
        f"GT samples: {gt_count} | GT triggers: {len(gt_triggers_s)} | "
        f"Predictions: {len(pred_scores)} | Eval pairs: {valid_pairs} | "
        f"Pred +: {pred_positive} | Pred -: {pred_negative} | "
        f"Stride: {stride} | Pack size: {pack_size} ({gt_shift_s * 1000.0:g}ms shift) | "
        f"Threshold: {threshold:g}"
    )
    if smooth_window_ms is not None and smooth_window_samples is not None:
        info_line_1 += (
            f" | Smooth: {smooth_window_ms:g}ms (~{smooth_window_samples} pred samples)"
            " | Classification on: all GT samples (linear interp on smoothed)"
        )
    else:
        info_line_1 += " | Classification on: all GT samples (linear interp on raw)"

    info_line_2 = (
        rf"$\bf{{Precision}}$: {precision_txt} | "
        rf"$\bf{{Recall}}$: {recall_txt} | "
        f"TP: {tp} FP: {fp} FN: {fn} TN: {tn}"
    )

    ax.text(
        0.99,
        1.09,
        info_line_1,
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
        ha="right",
    )
    ax.text(
        0.99,
        1.03,
        info_line_2,
        transform=ax.transAxes,
        fontsize=9,
        color="#222222",
        va="bottom",
        ha="right",
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.84])

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        click.echo(f"Figura salvata in: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
