"""
pipeline/snr_calculator.py
===========================
Compute transit Signal-to-Noise Ratio and combined significance.

Three methods combined:
  1. Phase-folded SNR  — depth / (per-point noise / sqrt(n_in_transit))
  2. TLS SDE           — Signal Detection Efficiency from TLS search
  3. TLS native SNR    — reported directly by TLS

Final SNR = weighted average: 0.5 × folded + 0.3 × TLS_SNR + 0.2 × SDE
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class SNRCalculator:

    def compute(self, time, flux, flux_err, tls_result) -> float:
        snr_fold = self._folded_snr(time, flux, flux_err, tls_result)
        sde      = float(tls_result.SDE or 0)
        tls_snr  = float(tls_result.snr or sde)

        snr = 0.50 * snr_fold + 0.30 * tls_snr + 0.20 * sde
        log.info(f"  SNR: folded={snr_fold:.2f}  TLS={tls_snr:.2f}  "
                 f"SDE={sde:.2f}  combined={snr:.2f}")
        return float(snr)

    def _folded_snr(self, time, flux, flux_err, r) -> float:
        try:
            period = float(r.period)
            t0     = float(r.T0)
            dur    = float(r.duration)
            depth  = float(r.depth)

            phase  = ((time - t0) % period) / period
            phase[phase > 0.5] -= 1.0
            half   = dur / period / 2.0
            in_tr  = np.abs(phase) < half
            out_tr = (np.abs(phase) > half * 2) & (np.abs(phase) < 0.4)

            if in_tr.sum() < 3 or out_tr.sum() < 5:
                return float(r.SDE or 0)

            noise = np.nanmedian(flux_err[out_tr]) \
                    if flux_err is not None else np.nanstd(flux[out_tr])
            return float(depth / (noise / np.sqrt(in_tr.sum()) + 1e-12))

        except Exception:
            return float(r.SDE or 0)

    @staticmethod
    def significance_statement(snr: float) -> str:
        if snr >= 20: return "Very high confidence (>5σ)"
        if snr >= 12: return "High confidence (~4σ)"
        if snr >= 9:  return "Moderate confidence (~3σ)"
        if snr >= 7:  return "Marginal detection (~2.5σ)"
        return               "Below threshold (<2σ)"
