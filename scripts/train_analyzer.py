#!/usr/bin/env python3
"""
Training-time analysis utilities for DeepStack model.

Provides per-epoch diagnostics:
- Per-street masked loss/MAE/RMSE
- Per-bucket correlation (summary stats)
- Calibration slice (error vs magnitude)
- CSV logs saved to models/reports and compact console summaries
"""
import os
import csv
import math
import json
from typing import Dict, Optional

import torch
import numpy as np


class TrainAnalyzer:
    def __init__(self, report_dir: str, scaling: Optional[Dict]=None):
        self.report_dir = report_dir
        os.makedirs(self.report_dir, exist_ok=True)
        self.scaling = scaling or {}
        self._csv_path = os.path.join(self.report_dir, 'training_metrics.csv')
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow([
                    'epoch','phase','loss','mae','rmse',
                    'corr_overall','corr_p1','corr_p2',
                    'pre_corr','flop_corr','turn_corr','river_corr'
                ])

    def _to_numpy(self, t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _de_std(self, y: torch.Tensor) -> torch.Tensor:
        if not self.scaling:
            return y
        mean = self.scaling.get('mean')
        std = self.scaling.get('std')
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            return y * std.view(1, -1) + mean.view(1, -1)
        return y

    def log_epoch(self, epoch: int, phase: str, outputs: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
                  streets: Optional[torch.Tensor] = None, loss_value: float = math.nan):
        """Compute diagnostics and append a CSV line; also saves JSON snapshots per epoch.
        """
        with torch.no_grad():
            # De-standardize for interpretability
            outputs_ds = self._de_std(outputs)
            targets_ds = self._de_std(targets)

            # Masked arrays for metrics
            mo = outputs * mask
            mt = targets * mask
            mae = torch.mean(torch.abs(mo - mt)).item()
            mse = torch.mean((mo - mt) ** 2).item()
            rmse = math.sqrt(mse)

            # Correlations on de-standardized values for human interpretability
            po = (outputs_ds * mask).flatten()
            pt = (targets_ds * mask).flatten()
            nz = (pt != 0) & (po != 0)
            corr_overall = float(np.corrcoef(self._to_numpy(po[nz]), self._to_numpy(pt[nz]))[0, 1]) if nz.any() else 0.0

            # Per-bucket correlation summary
            P = (outputs_ds * mask)
            T = (targets_ds * mask)
            num_out = P.shape[1]
            half = num_out // 2
            corrs = np.zeros(num_out, dtype=np.float32)
            Pn = self._to_numpy(P)
            Tn = self._to_numpy(T)
            for j in range(num_out):
                pj = Pn[:, j]
                tj = Tn[:, j]
                m = (pj != 0) & (tj != 0)
                if m.any():
                    corrs[j] = np.corrcoef(pj[m], tj[m])[0, 1]
            corr_p1 = float(np.nanmean(corrs[:half])) if half > 0 else 0.0
            corr_p2 = float(np.nanmean(corrs[half:])) if half > 0 else 0.0

            # Per-street correlations
            pre = flop = turn = river = float('nan')
            if streets is not None:
                s = streets.detach().cpu().numpy().astype(int)
                for street_id, varname in [(0,'pre'),(1,'flop'),(2,'turn'),(3,'river')]:
                    idx = s == street_id
                    if idx.any():
                        p = Pn[idx]
                        t = Tn[idx]
                        m = (p != 0) & (t != 0)
                        if m.any():
                            val = float(np.corrcoef(p[m], t[m])[0, 1])
                        else:
                            val = float('nan')
                    else:
                        val = float('nan')
                    if varname == 'pre': pre = val
                    elif varname == 'flop': flop = val
                    elif varname == 'turn': turn = val
                    elif varname == 'river': river = val

            # Append CSV
            with open(self._csv_path, 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow([epoch, phase, loss_value, mae, rmse, corr_overall, corr_p1, corr_p2, pre, flop, turn, river])

            # Save a compact JSON snapshot for the epoch (phase-specific)
            snap = {
                'epoch': epoch,
                'phase': phase,
                'loss': loss_value,
                'mae': mae,
                'rmse': rmse,
                'correlation': {
                    'overall': corr_overall,
                    'p1': corr_p1,
                    'p2': corr_p2,
                    'street': {'pre': pre, 'flop': flop, 'turn': turn, 'river': river}
                }
            }
            with open(os.path.join(self.report_dir, f'metrics_epoch_{epoch:03d}_{phase}.json'), 'w') as jf:
                json.dump(snap, jf, indent=2)

            # Console summary
            print(f"[{phase}] E{epoch:03d} loss={loss_value:.4f} mae={mae:.4f} rmse={rmse:.4f} corr={corr_overall:.3f} (p1={corr_p1:.3f}, p2={corr_p2:.3f}) streets pre={pre:.3f} flop={flop:.3f} turn={turn:.3f} river={river:.3f}")
