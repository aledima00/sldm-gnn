import csv
from pathlib import Path

import click
import matplotlib.pyplot as plt


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


def _load_gt_events(gt_csv: Path) -> tuple[int, list[float]]:
    """Load GT labels and return (n_rows, trigger_times_s)."""
    trigger_times_s: list[float] = []
    total_rows = 0

    with gt_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise click.ClickException("CSV ground truth senza header.")

        required = {"PackId", "MLBEncoded"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise click.ClickException(
                "CSV ground truth invalido: colonne mancanti "
                + ", ".join(sorted(missing))
            )

        for row_idx, row in enumerate(reader):
            total_rows += 1
            value_raw = (row.get("MLBEncoded") or "").strip()
            try:
                label_value = float(value_raw)
            except ValueError as exc:
                raise click.ClickException(
                    f"Valore MLBEncoded non numerico alla riga {row_idx + 2}: {value_raw!r}"
                ) from exc

            if label_value >= 0.5:
                time_s = (row_idx * SAMPLE_MS) / 1000.0
                trigger_times_s.append(time_s)

    return total_rows, trigger_times_s


def _load_predictions(pred_csv: Path) -> tuple[list[float], list[float]]:
    """Load prediction labels and scores from CSV."""
    scores: list[float] = []
    labels: list[float] = []

    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise click.ClickException("CSV predictions senza header.")

        required = {"PredictionLabels", "Scores"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise click.ClickException(
                "CSV predictions invalido: colonne mancanti "
                + ", ".join(sorted(missing))
            )

        for row_idx, row in enumerate(reader):
            label_raw = (row.get("PredictionLabels") or "").strip()
            value_raw = (row.get("Scores") or "").strip()

            try:
                label_value = float(label_raw)
            except ValueError as exc:
                raise click.ClickException(
                    f"Valore PredictionLabels non numerico alla riga {row_idx + 2}: {label_raw!r}"
                ) from exc

            try:
                score_value = float(value_raw)
            except ValueError as exc:
                raise click.ClickException(
                    f"Valore Scores non numerico alla riga {row_idx + 2}: {value_raw!r}"
                ) from exc

            labels.append(label_value)
            scores.append(score_value)

    return labels, scores


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--gt-csv",
    "gt_csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help='CSV ground truth con colonne ["PackId","MLBEncoded"].',
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
def main(
    gt_csv: Path,
    pred_csv: Path,
    stride: int,
    out_path: Path | None,
    show: bool,
    smooth_window_ms: float | None,
    pack_size: int,
) -> None:
    """Confronta temporalmente GT e prediction scores con densita diverse."""
    gt_count, gt_triggers_s = _load_gt_events(gt_csv)
    pred_labels, pred_scores = _load_predictions(pred_csv)

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

    ax.set_title("Ground Truth Trigger vs Prediction Score", loc="left")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Score")
    ax.set_xlim(left=0.0, right=max(gt_total_time_s, pred_total_time_s, 0.1))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    info_text = (
        f"GT samples: {gt_count} | GT triggers: {len(gt_triggers_s)} | "
        f"Predictions: {len(pred_scores)} | Pred labels positive: {sum(l >= 0.5 for l in pred_labels)} | "
        f"Stride: {stride} | Pack size: {pack_size} ({gt_shift_s * 1000.0:g}ms shift)"
    )
    if smooth_window_ms is not None and smooth_window_samples is not None:
        info_text += f" | Smooth window: {smooth_window_ms:g}ms (~{smooth_window_samples} pred samples)"
    ax.text(
        0.99,
        1.02,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
        ha="right",
    )

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])

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
