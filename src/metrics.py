import click as _click
from dataclasses import dataclass as _dc, field as _field
import numpy as _np
import pandas as _pd
from pathlib import Path as _Path
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from .utils import getLbName
from matplotlib import pyplot as _plt


@_dc
class EventMetrics:
    sim_duration_s: int
    threshold: float
    gt_arr_1d: _np.ndarray
    scr_arr_1d: _np.ndarray
    gap_pred: int = 5
    gap_gt: int = 20
    match_tol: int = 20

    preds_arr_1d: _np.ndarray = _field(init=False)
    gt_events: list[_np.ndarray] = _field(init=False)
    pred_clusters: list[_np.ndarray] = _field(init=False)

    gtec_tp_ids: set[int] = _field(init=False)
    pec_tp_ids: set[int] = _field(init=False)


    @staticmethod
    def __cluster(idx_array: _np.ndarray, gap: int) -> list[_np.ndarray]:
        """
        Given a sorted array of indices, cluster them into contiguous groups where consecutive indices differ by at most `gap`.
        Returns a list of numpy arrays, each containing the indices of a cluster.
        """
        if len(idx_array) == 0:
            return []
        clusters = [[idx_array[0]]]
        for i in range(1, len(idx_array)):
            if idx_array[i] - idx_array[i - 1] <= gap:
                clusters[-1].append(idx_array[i])
            else:
                clusters.append([idx_array[i]])
        return [_np.array(c) for c in clusters]

    def __post_init__(self):
        self.preds_arr_1d = (self.scr_arr_1d >= self.threshold).astype(_np.int32)

        gt_idx = _np.where(self.gt_arr_1d == 1)[0]
        self.gt_events = self.__cluster(gt_idx, gap=self.gap_gt)
        if not self.gt_events:
            _click.echo("  No GT events found, skipping event-level metrics.")
            return

        pred_idx = _np.where(self.preds_arr_1d == 1)[0]
        self.pred_clusters = self.__cluster(pred_idx, gap=self.gap_pred)

        self.gtec_tp_ids = set()
        self.pec_tp_ids = set()
        for ci, pc in enumerate(self.pred_clusters):
            pc_start, pc_end = pc[0], pc[-1]
            for ei, ge in enumerate(self.gt_events):
                gs, ge_end = ge[0], ge[-1]
                if pc_start <= ge_end + self.match_tol and pc_end >= gs - self.match_tol:
                    self.gtec_tp_ids.add(ei)
                    self.pec_tp_ids.add(ci)

    @property
    def n_gtevents(self) -> int:
        return len(self.gt_events)

    @property
    def n_pred_clusters(self) -> int:
        return len(self.pred_clusters)

    @property
    def n_detected_gte(self) -> int:
        return len(self.gtec_tp_ids)
    
    @property
    def n_missed_gte(self) -> int:
        return self.n_gtevents - self.n_detected_gte

    @property
    def n_tp_pred_clusters(self) -> int:
        return len(self.pec_tp_ids)
    
    @property
    def n_false_alarms(self) -> int:
        return self.n_pred_clusters - self.n_tp_pred_clusters
    
    @property
    def far_h(self) -> float:
        return (self.n_false_alarms / self.sim_duration_s) * 3600

    @property
    def event_precision(self) -> float:
        return self.n_tp_pred_clusters / self.n_pred_clusters if self.n_pred_clusters > 0 else 0.0

    @property
    def event_recall(self) -> float:
        return self.n_detected_gte / self.n_gtevents if self.n_gtevents > 0 else 0.0

    def printout(self):
        _click.echo(f"\nEVENT LEVEL METRICS:\n-- Generic Config:")
        _click.echo(f"   Simulation duration (s): {self.sim_duration_s}")
        _click.echo(f"   Threshold: {self.threshold}")
        _click.echo(f"   Gap for clustering predictions: {self.gap_pred} samples")
        _click.echo(f"   Gap for clustering GT events: {self.gap_gt} samples")
        _click.echo(f"   Match tolerance for detected events: {self.match_tol} samples")
        _click.echo(f"-- GT events ({self.n_gtevents})")
        _click.echo(f"   Detected GT events: {self.n_detected_gte}/{self.n_gtevents}")
        _click.echo(f"   Missed GT events: {self.n_missed_gte}/{self.n_gtevents}")
        _click.echo(f"-- Predicted clusters ({self.n_pred_clusters})")
        _click.echo(f"   True positive clusters: {self.n_tp_pred_clusters}/{self.n_pred_clusters}")
        _click.echo(f"   False alarm clusters: {self.n_false_alarms}/{self.n_pred_clusters}")
        _click.echo(f"   False alarm rate per hour: {self.far_h:.4f}")
        _click.echo(f"-- Event-level stats:")
        _click.echo(f"   Event precision: {self.event_precision:.4f}")
        _click.echo(f"   Event recall: {self.event_recall:.4f}")

    def plot_in_csv(self,outdir:_Path, lb_value:int):
        event_rows = [{
            'label': lb_value,
            'label_name': getLbName(lb_value),
            'threshold': self.threshold,
            'sim_duration_s': self.sim_duration_s,
            'n_gt_events': self.n_gtevents,
            'n_detected_gt_events': self.n_detected_gte,
            'n_missed_gt_events': self.n_missed_gte,
            'n_pred_clusters': self.n_pred_clusters,
            'n_tp_pred_clusters': self.n_tp_pred_clusters,
            'n_false_alarm_clusters': self.n_false_alarms,
            'far_per_hour': self.far_h,
            'event_precision': self.event_precision,
            'event_recall': self.event_recall,
        }]
        _pd.DataFrame(event_rows).to_csv(outdir / 'test_event_metrics.csv', index=False)
        _click.echo(f"Saved to {outdir / 'test_event_metrics.csv'}")
    
    def plot_temporal_comparison(self, outpath: _Path):
        fig, (ax, ax_detail) = _plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [3, 1]})

        x_axis = _np.arange(len(self.scr_arr_1d))
        ax.plot(x_axis, self.scr_arr_1d, color='#4a4abc', linewidth=1.2, alpha=0.85, label='Score')

        for idx in _np.where(self.gt_arr_1d == 1)[0]:
            ax.axvline(x=idx, color='red', alpha=0.4, linewidth=1.7, linestyle='-')

        ax.axhline(y=self.threshold, color='green', linewidth=1.5, linestyle='--', alpha=0.9,
                    label=f'Threshold ({self.threshold:g})')

        for idx,p in enumerate(self.pred_clusters):
            color = '#22aa44' if idx in self.pec_tp_ids else '#dd6622'
            alpha = 0.2 if idx in self.pec_tp_ids else 0.15
            ax.axvspan(p[0], p[-1], alpha=alpha, color=color, linewidth=0)

        ax.set_title(f'Score vs Ground Truth Events', loc='left',
                        fontsize=11, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(bottom=-0.05, top=1.05)
        ax.grid(True, alpha=0.25)

        legend_elements = [
            _plt.Line2D([0], [0], color='#4a4abc', linewidth=1.5, label='Score'),
            _plt.Line2D([0], [0], color='red', linewidth=1.5, linestyle='-', label='GT event'),
            _plt.Line2D([0], [0], color='green', linewidth=1.5, linestyle='--', label=f'Threshold ({self.threshold:g})'),
            _plt.Rectangle((0, 0), 1, 1, color='#22aa44', alpha=0.25, label='Detected (TP)'),
            _plt.Rectangle((0, 0), 1, 1, color='#dd6622', alpha=0.25, label='False alarm'),
        ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        #  -> indications text box

        n_gtp_frames = int((self.gt_arr_1d == 1).sum())
        n_predp_frames = int((self.preds_arr_1d == 1).sum())

        info_text = (
            f"Samples: {len(self.scr_arr_1d)} | GT frames: {n_gtp_frames} | GT event clusters: {self.n_gtevents} | "
            f"Pred +: {n_predp_frames} | Threshold: {self.threshold:g}"
        )
        if self.pred_clusters is not None:
            info_text += (
                f"\nPrediction clusters: {self.n_pred_clusters} | "
                f"Detected: {self.n_detected_gte}/{self.n_gtevents} | "
                f"False alarm clusters: {self.n_false_alarms}"
            )
        ax.text(0.99, 1.07, info_text, transform=ax.transAxes, fontsize=9,
                color='#444444', va='bottom', ha='right')

        ## todo

        ax_detail.bar(x_axis, self.preds_arr_1d, color=['#22aa44' if self.gt_arr_1d[i] else '#dd6622' for i in range(len(self.preds_arr_1d))],
                        width=1.0, linewidth=0)
        ax_detail.set_xlabel('Sample Index')
        ax_detail.set_ylabel('Prediction')
        ax_detail.set_yticks([0, 1])
        ax_detail.set_yticklabels(['0', '1'])
        ax_detail.set_ylim(bottom=-0.1, top=1.1)
        ax_detail.grid(True, alpha=0.15, axis='y')

        _plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
        fig.savefig(outpath, dpi=150)
        _plt.close(fig)

@_dc
class PackMetrics:
    gt_arr_1d: _np.ndarray
    scr_arr_1d: _np.ndarray
    threshold: float

    pred_arr_1d: _np.ndarray = _field(init=False)
    conf_matrix: _np.ndarray = _field(init=False)
    precision: float = _field(init=False)
    recall: float = _field(init=False)
    f1_score: float = _field(init=False)

    accuracy: float = _field(init=False)
    roc_auc: float = _field(init=False)
    ap: float = _field(init=False)

    

    def __post_init__(self):
        self.pred_arr_1d = (self.scr_arr_1d >= self.threshold).astype(_np.int32)
        self.conf_matrix = confusion_matrix(self.gt_arr_1d, self.pred_arr_1d, labels=[0, 1])
        self.precision, self.recall, self.f1_score, _ = precision_recall_fscore_support(self.gt_arr_1d, self.pred_arr_1d, average='binary', zero_division=0)

        self.accuracy = float((self.pred_arr_1d == self.gt_arr_1d).mean())
        self.roc_auc = float(roc_auc_score(self.gt_arr_1d, self.scr_arr_1d)) if _np.unique(self.gt_arr_1d).size > 1 else _np.nan
        self.ap = float(average_precision_score(self.gt_arr_1d, self.scr_arr_1d)) if _np.unique(self.gt_arr_1d).size > 1 else _np.nan
    
    def printout(self):
        _click.echo(f"\nPACK-LEVEL METRICS:")
        _click.echo(f"  Label [value {self.gt_arr_1d[0]}]: {getLbName(self.gt_arr_1d[0])}")
        _click.echo(f"  Threshold: {self.threshold}")
        _click.echo(f"  Accuracy: {self.accuracy:.4f}")
        _click.echo(f"  Precision: {self.precision:.4f}")
        _click.echo(f"  Recall: {self.recall:.4f}")
        _click.echo(f"  F1 Score: {self.f1_score:.4f}")
        _click.echo(f"  ROC AUC: {self.roc_auc:.4f}")
        _click.echo(f"  Average Precision: {self.ap:.4f}")
        _click.echo(f"  Confusion Matrix (TN, FP, FN, TP): {self.conf_matrix.ravel().tolist()}")
        _click.echo(f"  Num Samples: {self.gt_arr_1d.size}")

    def plot_in_csv(self, outdir: _Path, lb_value):
        metrics_rows = [{
            'label': lb_value,
            'label_name': getLbName(lb_value),
            'threshold': self.threshold,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'roc_auc': self.roc_auc,
            'average_precision': self.ap,
            'tn': self.conf_matrix[0, 0],
            'fp': self.conf_matrix[0, 1],
            'fn': self.conf_matrix[1, 0],
            'tp': self.conf_matrix[1, 1],
            'num_samples': self.gt_arr_1d.size
        }]
        _pd.DataFrame(metrics_rows).to_csv(outdir / 'test_pack_metrics.csv', index=False)
        _click.echo(f"Saved to {outdir / 'test_pack_metrics.csv'}")