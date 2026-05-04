import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch_geometric.loader import DataLoader as GDL

import src.transforms as TFs
from src.dataset import MapGraph
from src.models.grusage import GruSage
from src.utils import MetaData, getLbName


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
def main(inputdir: Path, outdir: Path, weights_path: Path, batch_size: int, threshold: float, cut: int | None):
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

        metrics_rows.append({
            'ActiveLabelIndex': local_lb_idx,
            'GlobalLabelId': global_lb,
            'LabelName': getLbName(local_lb_idx, active_labels),
            'Accuracy': accuracy,
            'Precision': float(precision),
            'Recall': float(recall),
            'F1': float(f1),
            'ROCAUC': roc_auc,
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'NumSamples': int(gt.shape[0]),
        })

    details_df = pd.DataFrame(details_rows)
    metrics_df = pd.DataFrame(metrics_rows)

    details_path = outpath / 'test_predictions_vs_gt.csv'
    metrics_path = outpath / 'test_metrics_summary.csv'
    details_df.to_csv(details_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    overall_accuracy = float((preds_all == gt_all).mean())
    click.echo(f"Overall multilabel accuracy: {overall_accuracy:.4f}")
    click.echo(f"Saved detailed predictions vs gt: {details_path}")
    click.echo(f"Saved metrics summary: {metrics_path}")


if __name__ == '__main__':
    main()