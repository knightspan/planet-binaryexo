"""
pipeline/detector.py
=====================
Transit search using Transit Least Squares (TLS) — Hippke & Heller (2019).

TLS advantages over Box Least Squares:
  - Uses physically accurate limb-darkened transit shape (not a box)
  - ~10–15% better sensitivity to small planets
  - Provides SDE (Signal Detection Efficiency) as significance metric
  - Period uncertainty from SDE peak width
  - Correctly models ingress/egress

Falls back to astropy BLS if TLS is not installed.
Progress bars suppressed for clean logging.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class TLSResult:
    """Unified result container regardless of TLS or BLS backend."""
    __slots__ = [
        'period', 'period_uncertainty', 'T0', 'depth', 'duration', 'SDE',
        'snr', 'transit_times', 'transit_count', 'distinct_transit_count',
        'odd_even_mismatch', 'periods', 'power', 'folded_phase', 'folded_y',
        'model_folded_phase', 'model_folded_model',
    ]
    def __init__(self):
        for s in self.__slots__:
            setattr(self, s, None)


class TransitDetector:

    def __init__(self, config: dict):
        self.period_min   = config.get('period_min',           0.5)
        self.period_max   = config.get('period_max',          13.0)
        self.min_transits = config.get('min_transit_count',     3)
        # Fast mode: lower oversampling for demos/testing (less precise period,
        # ~3-5x faster). Default oversampling_factor=5 is full ISRO-submission
        # quality and should be used for the final report.
        self.oversampling = config.get('tls_oversampling_factor', 5)

    def search(self, time, flux, flux_err):
        """Run TLS (preferred) or BLS (fallback). Returns TLSResult or None."""
        try:
            return self._tls(time, flux, flux_err)
        except ImportError:
            log.warning("  transitleastsquares not installed — falling back to BLS")
            return self._bls(time, flux, flux_err)
        except Exception as e:
            log.warning(f"  TLS failed ({e}) — falling back to BLS")
            return self._bls(time, flux, flux_err)

    # ─────────────────────────────────────────────────────────────────────────
    def _tls(self, time, flux, flux_err):
        from transitleastsquares import transitleastsquares
        log.info(f"  Running TLS  period=[{self.period_min},{self.period_max}]d")

        model = transitleastsquares(time, flux, flux_err)
        raw   = model.power(
            show_progress_bar    = False,     # suppress verbose output
            period_min           = self.period_min,
            period_max           = self.period_max,
            n_transits_min       = self.min_transits,
            u                    = [0.40, 0.26],  # quadratic LD, TESS G-band
            oversampling_factor  = self.oversampling,
            duration_grid_step   = 1.1,
        )

        r = TLSResult()
        r.period             = float(raw.period)
        r.period_uncertainty = float(raw.period_uncertainty)
        r.T0                 = float(raw.T0)
        # TLS reports `depth` as the relative flux AT transit bottom
        # (close to 1.0, e.g. 0.998), not the fractional depth itself.
        # True transit depth = 1 - flux_at_bottom.
        raw_depth_value      = float(raw.depth)
        r.depth              = float(1.0 - raw_depth_value) \
                                if raw_depth_value > 0.5 else float(raw_depth_value)
        r.duration           = float(raw.duration)
        r.SDE                = float(raw.SDE)
        r.snr                = float(raw.snr) if hasattr(raw, 'snr') else r.SDE
        r.transit_times      = list(raw.transit_times) \
                                if hasattr(raw, 'transit_times') else []
        r.transit_count      = int(raw.transit_count) \
                                if hasattr(raw, 'transit_count') else 0
        r.distinct_transit_count = int(raw.distinct_transit_count) \
                                    if hasattr(raw, 'distinct_transit_count') \
                                    else r.transit_count
        r.odd_even_mismatch  = float(raw.odd_even_mismatch) \
                                if hasattr(raw, 'odd_even_mismatch') else 1.0
        r.periods            = np.array(raw.periods) \
                                if hasattr(raw, 'periods') else None
        r.power              = np.array(raw.power) \
                                if hasattr(raw, 'power') else None
        r.folded_phase       = np.array(raw.folded_phase) \
                                if hasattr(raw, 'folded_phase') else None
        r.folded_y           = np.array(raw.folded_y) \
                                if hasattr(raw, 'folded_y') else None
        r.model_folded_phase = np.array(raw.model_folded_phase) \
                                if hasattr(raw, 'model_folded_phase') else None
        r.model_folded_model = np.array(raw.model_folded_model) \
                                if hasattr(raw, 'model_folded_model') else None

        log.info(f"  TLS best period={r.period:.5f}d  "
                 f"depth={r.depth*100:.4f}%  "
                 f"dur={r.duration*24:.2f}h  SDE={r.SDE:.2f}")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    def _bls(self, time, flux, flux_err):
        from astropy.timeseries import BoxLeastSquares
        import astropy.units as u
        log.info(f"  Running BLS (fallback)  period=[{self.period_min},{self.period_max}]d")

        periods = np.linspace(self.period_min, self.period_max, 10000)
        bls     = BoxLeastSquares(time * u.day, flux, dy=flux_err)
        pg      = bls.power(
            period   = periods * u.day,
            duration = [0.05, 0.10, 0.15, 0.20] * u.day,
            objective = 'snr'
        )
        best = np.argmax(pg.power)

        r = TLSResult()
        r.period             = float(pg.period[best].value)
        r.period_uncertainty = r.period * 0.001
        r.T0                 = float(pg.transit_time[best].value)
        r.depth              = float(pg.depth[best])
        r.duration           = float(pg.duration[best].value)
        r.SDE                = float(pg.power[best])
        r.snr                = r.SDE
        r.transit_times      = []
        r.transit_count      = max(1, int(27.0 / r.period))
        r.distinct_transit_count = r.transit_count
        r.odd_even_mismatch  = 1.0
        r.periods            = periods
        r.power              = np.array(pg.power)
        r.folded_phase       = None
        r.folded_y           = None
        r.model_folded_phase = None
        r.model_folded_model = None

        log.info(f"  BLS best period={r.period:.4f}d  "
                 f"depth={r.depth*100:.4f}%  SNR={r.snr:.2f}")
        return r

    # ─────────────────────────────────────────────────────────────────────────
    def search_secondary(self, time, flux, tls_result):
        """Search for secondary eclipse at phase 0.5 (EB diagnostic)."""
        if tls_result is None or tls_result.period is None:
            return 0.0, 0.0
        period = float(tls_result.period)
        t0     = float(tls_result.T0)
        dur    = float(tls_result.duration)

        phase = ((time - (t0 + period / 2)) % period) / period
        phase[phase > 0.5] -= 1.0
        window  = (dur / period) * 1.5
        in_sec  = np.abs(phase) < window
        out_sec = np.abs(phase) > window * 3

        if in_sec.sum() < 3 or out_sec.sum() < 10:
            return 0.0, 0.0

        baseline = np.nanmedian(flux[out_sec])
        depth    = max(float(baseline - np.nanmin(flux[in_sec])), 0.0)
        noise    = np.nanstd(flux[out_sec]) / np.sqrt(in_sec.sum())
        snr      = depth / max(noise, 1e-12)
        return depth, float(snr)
