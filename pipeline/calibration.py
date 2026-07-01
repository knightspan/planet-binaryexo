"""
pipeline/calibration.py
========================
Temperature scaling for CNN confidence calibration.

Cross-entropy-trained CNNs are systematically overconfident (Guo et al. 2017).
A raw softmax output of 0.92 does NOT mean 92% of such predictions are correct.
Temperature scaling fits a single scalar T on a held-out validation set:

  p_calibrated = softmax(logits / T)

T > 1 softens (de-confidences) the distribution.
T is fit by minimizing negative log-likelihood on the validation set.
Accuracy is UNCHANGED by T — only the confidence values are adjusted.

Output: calibration_temperature.json with T and a reliability table.
The reliability table is the evidence that calibration was actually checked:
  each bin shows predicted confidence vs observed accuracy in that bin.
  A well-calibrated model has matching values in each bin.
"""

import numpy as np
import logging
from scipy.optimize import minimize_scalar

log = logging.getLogger(__name__)


def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


def _nll(T, logits, y_true):
    """Negative log-likelihood under temperature T."""
    T     = max(T, 1e-3)
    probs = _softmax(logits / T)
    probs = np.clip(probs, 1e-12, 1.0)
    return -np.mean(np.log(probs[np.arange(len(y_true)), y_true.astype(int)]))


class ConfidenceCalibrator:
    """Post-hoc temperature scaling calibrator."""

    def __init__(self):
        self.temperature = 1.0
        self.fitted      = False

    def fit(self, val_logits, val_labels):
        """
        val_logits : (N, n_classes) pre-softmax outputs on validation set
        val_labels : (N,) true class indices
        """
        result           = minimize_scalar(
            _nll, bounds=(0.05, 10.0), method='bounded',
            args=(val_logits, val_labels))
        self.temperature = float(result.x)
        self.fitted      = True
        log.info(f"  Temperature scaling: T={self.temperature:.3f} "
                 f"({'softening' if self.temperature > 1 else 'sharpening'})")
        return self.temperature

    def predict_calibrated(self, logits):
        T = self.temperature if self.fitted else 1.0
        return _softmax(logits / T)

    def reliability_table(self, val_logits, val_labels, n_bins=10):
        """
        Returns a list of dicts, one per confidence bin, showing:
          mean_confidence vs observed_accuracy.
        A well-calibrated model has these matching.
        """
        probs   = self.predict_calibrated(val_logits)
        preds   = np.argmax(probs, axis=1)
        confs   = np.max(probs, axis=1)
        correct = (preds == val_labels.astype(int)).astype(float)

        bins = np.linspace(0, 1, n_bins + 1)
        rows = []
        for i in range(n_bins):
            mask = (confs >= bins[i]) & (confs < bins[i + 1])
            if mask.sum() == 0:
                continue
            rows.append({
                'bin_range':         (round(float(bins[i]), 2),
                                      round(float(bins[i+1]), 2)),
                'n_samples':         int(mask.sum()),
                'mean_confidence':   float(confs[mask].mean()),
                'observed_accuracy': float(correct[mask].mean()),
            })
        return rows

    def save(self, path='models/calibration_temperature.json'):
        import json
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'temperature': self.temperature, 'fitted': self.fitted}, f)

    def load(self, path='models/calibration_temperature.json'):
        import json
        try:
            with open(path) as f:
                d = json.load(f)
            self.temperature = d.get('temperature', 1.0)
            self.fitted      = True
        except Exception:
            pass
