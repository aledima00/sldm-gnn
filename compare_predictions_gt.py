import csv
from pathlib import Path

import click
import numpy as _np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from src.metrics import EventMetrics, PackMetrics
from src.utils import bayesPriorShift

from typing import Literal as Lit, get_args

on_empty_options = Lit['drop', 'zero']

def _load_gt_events(gt_parquet: Path, active_label: int | None):
    gts_list: list[int] = []

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
        gts_list.append(gt_bin)

    return _np.array(gts_list, dtype=_np.int32)


def _load_prediction_scores(pred_csv: Path, on_empty: on_empty_options = 'drop') -> _np.ndarray:
    scores: list[float] = []
    with pred_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise click.ClickException("CSV predictions senza header.")

        if "Scores" in reader.fieldnames:
            score_field = "Scores"
        elif "Score" in reader.fieldnames:
            score_field = "Score"
        else:
            raise click.ClickException(
                "CSV predictions invalido: colonne mancanti Scores/Score"
            )

        for row in reader:
            value_raw = (row.get(score_field) or "").strip()
            if value_raw in {".", ""}:
                # "." or "" represents empty samples (e.g. no vehicles), where the model is not fed with anything;
                match on_empty:
                    case "drop":
                        continue  # skip this sample entirely (removing it from scores array)
                    case "zero":
                        scores.append(0.0)  # set score to 0 (keeps alignment
                    case _:
                        raise ValueError(f"Invalid value for 'on_empty': {on_empty}")
            else:
                try:
                    scores.append(float(value_raw))
                except ValueError:
                    scores.append(_np.nan)
    return _np.array(scores, dtype=_np.float32)


@click.command()
@click.option("--gt-parquet", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help='Parquet ground truth con colonne ["PackId","MLBEncoded"].')
@click.option("--pred-csv", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="CSV predictions (output of rcv.py) con colonne [PredictionLabels,Scores].")
@click.option("--threshold", type=click.FloatRange(min=0.0, max=1.0), default=0.5, show_default=True, help="Soglia per classificare i prediction scores in positivi/negativi.")
@click.option("--outdir", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=None, help="Directory di output per metriche, plot e CSV.")
@click.option("-e", "--event-metrics", is_flag=True, default=False, help="Compute event-level clustering metrics (FAR/h, event precision).")
@click.option("--sim-duration", type=int, default=60, show_default=True, help="Simulation duration in seconds (for FAR/h computation).")
@click.option("--active-label", type=int, default=None, help="Active label index to extract from MLBEncoded bitmask (e.g. 2 for TURN).")
@click.option("--calibrate-priors", is_flag=True, default=False, help="Apply Bayes prior-shift calibration. Requires --train-prior and --test-prior.")
@click.option("--train-prior", type=float, default=None, help="Training P(y=1) for prior-shift calibration (e.g. 0.4636 for f2_turn).")
@click.option("--test-prior", type=float, default=None, help="Deployment P(y=1) for prior-shift calibration (e.g. 0.00356 for f2_turn).")
@click.option("--nan-policy", type=click.Choice(["drop", "zero"], case_sensitive=False), default="zero", show_default=True, help="How to handle missing/invalid scores: drop samples or set score to 0.")
@click.option('--gap-pred', type=int, default=5, show_default=True, help='Gap (samples) for clustering prediction indices.')
@click.option('--gap-gt', type=int, default=20, show_default=True, help='Gap (samples) for clustering ground-truth indices.')
@click.option('--match-tol', type=int, default=20, show_default=True, help='Tolerance (samples) when matching predicted clusters to GT events.')
@click.option('--on-empty',"on_empty", type=click.Choice(get_args(on_empty_options), case_sensitive=False), default='drop', show_default=True, help='How to handle empty samples (e.g. no vehicles in the map). drop: remove from scores array (may be used when we want to evaluate only on non-empty samples); zero: set score to 0 (keeps alignment with GT, may be used when we want to evaluate on all samples including empty ones).')
def main(gt_parquet, pred_csv, threshold, outdir, event_metrics, sim_duration, active_label, calibrate_priors, train_prior, test_prior, nan_policy, gap_pred, gap_gt, match_tol, on_empty):
    """Confronta temporalmente GT e prediction scores con metriche event-level."""
    if calibrate_priors:
        if train_prior is None or test_prior is None:
            raise click.ClickException(
                "--calibrate-priors requires --train-prior and --test-prior"
            )

    if outdir is None:
        outdir = Path.cwd()
    outdir.mkdir(parents=True, exist_ok=True)

    gts = _load_gt_events(gt_parquet, active_label)
    scores = _load_prediction_scores(pred_csv, on_empty=on_empty)

    if gts.size > scores.size:
        raise click.ClickException(
            "GT array longer than prediction scores array. An error during inference may be occurred."
            f"(gt={gts.size}, pred={scores.size})."
        )
    else:
        if gts.size < scores.size:
            click.echo(f"Warning: more prediction scores ({scores.size}) than GT samples ({gts.size}). Truncating predictions to match GT length.")
            scores = scores[:gts.size]
        click.echo(f"Loaded GT and predictions: {gts.size} samples.")
        

    valid_mask = ~_np.isnan(scores)
    if not valid_mask.all():
        dropped = int((~valid_mask).sum())
        if nan_policy == "zero":
            click.echo(f"Replacing {dropped} missing/invalid scores with 0.0 to keep alignment.")
            scores = _np.nan_to_num(scores, nan=0.0)
        else:
            raise click.ClickException(f"Unimplemented NaN policy: {nan_policy}. Consider using --nan-policy zero to replace missing scores with 0.")

    if calibrate_priors:
        scores, prior_ratio = bayesPriorShift(scores, train_prior, test_prior)
        click.echo(f"Prior-shift calibration: train P(y=1)={train_prior:.6f}, test P(y=1)={test_prior:.6f}")
        click.echo(f"  Prior ratio: {prior_ratio:.6f}")

    ## ==================== COMPUTE METRICS ====================

    pm = PackMetrics(
        gt_arr_1d=gts,
        scr_arr_1d=scores,
        threshold=threshold
    )
    pm.printout()
    pm.plot_in_csv(outdir, lb_value=active_label)


    if event_metrics:
        em = EventMetrics(
            sim_duration_s = sim_duration,
            threshold = threshold,
            gt_arr_1d = gts,
            scr_arr_1d = scores,
            gap_pred = gap_pred,
            gap_gt = gap_gt,
            match_tol = match_tol,
        )
        em.printout()
        em.plot_in_csv(outdir, lb_value=active_label)
        em.plot_temporal_comparison(outdir / f"test_temporal_plot_lb{active_label}.png")


if __name__ == "__main__":
    main()
