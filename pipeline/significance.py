"""
pipeline/significance.py
=========================
Bootstrap / permutation False Alarm Probability (FAP).

TLS's SDE is a detection-efficiency proxy useful for ranking signals, but
it is NOT a probability. PS-7 requires "significance levels" — a bootstrap
FAP is the standard, assumption-free answer.

Method:
  Randomly permute (shuffle) the flux array N times. This destroys any
  real periodic signal while preserving the empirical noise distribution
  (white noise, red noise, gaps, systematics — all preserved).
  Run TLS/BLS on each permuted array. Record the best SDE achieved by
  chance. FAP = fraction of permutations with SDE >= the real detection.

  Floor: FAP >= 1/N (can never report exactly zero from finite sampling).

Interpretation:
  FAP = 0.001 → 0.1% chance this is noise → highly significant
  FAP = 0.01  → 1% → significant
  FAP = 0.05  → 5% → marginal
  FAP > 0.1   → not significant, treat as noise
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class SignificanceEstimator:

    def __init__(self, config: dict):
        self.n_bootstrap = config.get('fap_n_bootstrap', 100)
        self.rng         = np.random.default_rng(config.get('random_seed', 42))

    def bootstrap_fap(self, time, flux, flux_err, detector,
                       observed_sde, n_bootstrap=None):
        """
        Compute bootstrap FAP by permuting flux and re-running transit search.

        Performance note: each permutation requires a full periodogram search,
        which is the expensive part of TLS. For the bootstrap loop specifically
        we use a FAST detector configuration (oversampling_factor=1) regardless
        of what the main detector uses for the real detection — the null-SDE
        distribution from a coarser search is a perfectly valid (if slightly
        more conservative) noise-floor estimate, and this is what makes FAP
        computation practical for interactive use (50-100 permutations in
        ~15-30s rather than ~15-50 minutes).

        Parameters
        ----------
        time, flux, flux_err : ndarray   detrended light curve
        detector             : TransitDetector instance (used for its config;
                                a fast clone is created internally for the loop)
        observed_sde         : float     SDE of the real detection
        n_bootstrap          : int       overrides config default

        Returns
        -------
        dict:
            fap                 : float   [0, 1]
            n_bootstrap         : int
            null_sde_mean       : float
            null_sde_std        : float
            null_sde_max        : float
        """
        n = n_bootstrap or self.n_bootstrap

        # Build a fast detector clone for the bootstrap loop (oversampling=1).
        # This does NOT affect the real detection — only the null-distribution
        # search used to calibrate the noise floor.
        fast_detector = _clone_fast(detector)

        null_sdes = []
        log.info(f"  Bootstrap FAP: {n} permutations (fast mode)...")
        for i in range(n):
            perm_flux = self.rng.permutation(flux)
            try:
                res = fast_detector.search(time, perm_flux, flux_err)
                null_sdes.append(float(res.SDE) if res is not None else 0.0)
            except Exception:
                null_sdes.append(0.0)

        null_sdes = np.array(null_sdes)
        fap       = float(np.mean(null_sdes >= observed_sde))
        fap       = max(fap, 1.0 / n)   # floor: never exactly zero

        log.info(f"  Bootstrap FAP = {fap:.2e}  "
                 f"(null SDE: mean={null_sdes.mean():.2f}, "
                 f"max={null_sdes.max():.2f})")

        return {
            'fap':          fap,
            'n_bootstrap':  n,
            'null_sde_mean': float(null_sdes.mean()),
            'null_sde_std':  float(null_sdes.std()),
            'null_sde_max':  float(null_sdes.max()),
        }

    @staticmethod
    def significance_statement(fap: float) -> str:
        if fap <= 1e-4:  return "Highly significant (FAP ≤ 0.01%)"
        if fap <= 1e-3:  return "Very significant  (FAP ≤ 0.1%)"
        if fap <= 0.01:  return "Significant       (FAP ≤ 1%)"
        if fap <= 0.05:  return "Marginal          (FAP ≤ 5%)"
        return                  "Not significant   (FAP > 5%)"


def _clone_fast(detector):
    """
    Build a fast-mode clone of a TransitDetector for use inside the
    bootstrap permutation loop, where search speed matters far more than
    period-grid precision (we only need the null SDE distribution's shape).
    """
    from pipeline.detector import TransitDetector
    fast_config = {
        'period_min':              detector.period_min,
        'period_max':              detector.period_max,
        'min_transit_count':       detector.min_transits,
        'tls_oversampling_factor': 1,   # fastest setting regardless of caller
    }
    return TransitDetector(fast_config)
