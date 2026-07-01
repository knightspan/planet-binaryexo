"""
pipeline/preprocessor.py
=========================
Light curve preprocessing using WOTAN robust biweight flattening.

Why Wotan over Savitzky-Golay:
  - Biweight estimator is resistant to outliers (not affected by transits themselves)
  - No polynomial ringing artifacts that create spurious dips
  - Handles variable stellar activity through robust local statistics
  - break_tolerance parameter correctly handles momentum dump gaps

Pipeline:
  1. Unpack lightkurve object OR dict (synthetic)
  2. Remove NaN / Inf
  3. Sort by time
  4. Normalise flux to median = 1.0
  5. First sigma clip (catch cosmic rays, +7σ)
  6. Wotan biweight detrend (window=0.75d default)
  7. Second sigma clip on residuals (5σ)
  8. Return (time, flux, flux_err) numpy float64 arrays
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class LightCurvePreprocessor:

    def __init__(self, config: dict):
        self.sigma_clip   = config.get('sigma_clip',   5.0)
        self.wotan_window = config.get('wotan_window', 0.75)
        self.wotan_method = config.get('wotan_method', 'biweight')

    def process(self, lc_input):
        """
        Accept a lightkurve LightCurve object or a synthetic dict.
        Returns (time, flux, flux_err) as float64 numpy arrays.
        """
        time, flux, flux_err = self._unpack(lc_input)
        n_raw = len(time)

        # Sort
        idx      = np.argsort(time)
        time     = time[idx]
        flux     = flux[idx]
        flux_err = flux_err[idx]

        # Normalise
        med      = np.nanmedian(flux)
        flux     = flux / med
        flux_err = flux_err / max(med, 1e-12)

        # First clip: remove +7σ outliers (cosmic rays, saturation)
        time, flux, flux_err = self._clip(time, flux, flux_err, sigma=7.0)

        # Detrend
        flat_flux, trend = self._detrend(time, flux)
        flat_err         = flux_err / np.abs(trend + 1e-12)

        # Second clip: residual outliers
        time, flat_flux, flat_err = self._clip(
            time, flat_flux, flat_err, sigma=self.sigma_clip)

        n_out = len(time)
        log.info(f"  After preprocessing: {n_out} points "
                 f"({n_out/n_raw*100:.1f}% retained)")
        log.info(f"  Median flux err: {np.nanmedian(flat_err)*1e6:.1f} ppm")

        return time, flat_flux, flat_err

    def _unpack(self, lc):
        if isinstance(lc, dict):
            t = np.array(lc['time'],     dtype=np.float64)
            f = np.array(lc['flux'],     dtype=np.float64)
            e = np.array(lc['flux_err'], dtype=np.float64)
        else:
            # lightkurve LightCurve or LightCurveCollection
            try:
                from lightkurve import LightCurveCollection
                if isinstance(lc, LightCurveCollection):
                    lc = lc.stitch()
            except Exception:
                pass
            t = lc.time.value.astype(np.float64)
            f = lc.flux.value.astype(np.float64)
            try:
                e = lc.flux_err.value.astype(np.float64)
            except Exception:
                e = np.full_like(f, np.nanstd(f) * 0.1)

        # Remove NaN/Inf
        ok = np.isfinite(t) & np.isfinite(f) & np.isfinite(e)
        return t[ok], f[ok], e[ok]

    def _detrend(self, time, flux):
        try:
            from wotan import flatten
            flat_flux, trend = flatten(
                time, flux,
                method        = self.wotan_method,
                window_length = self.wotan_window,
                return_trend  = True,
                break_tolerance = 0.5,
                cval          = 5.0,
            )
            log.info(f"  Wotan '{self.wotan_method}' detrend applied "
                     f"(window={self.wotan_window}d)")
            # Fill any NaN left by wotan at edges
            nan_mask = ~np.isfinite(flat_flux)
            if nan_mask.sum() > 0:
                flat_flux[nan_mask] = 1.0
                trend[nan_mask]     = np.nanmedian(trend)
            return flat_flux, trend

        except ImportError:
            log.warning("  wotan not installed — using Savitzky-Golay fallback")
            return self._sg_fallback(time, flux)
        except Exception as e:
            log.warning(f"  Wotan failed ({e}) — Savitzky-Golay fallback")
            return self._sg_fallback(time, flux)

    def _sg_fallback(self, time, flux):
        from scipy.signal import savgol_filter
        dt  = np.nanmedian(np.diff(time))
        win = int(round(1.0 / dt))
        win = win + (1 - win % 2)   # ensure odd
        win = max(win, 11)
        trend     = savgol_filter(flux, window_length=win, polyorder=2)
        flat_flux = flux / trend
        return flat_flux, trend

    def _clip(self, time, flux, flux_err, sigma=5.0, max_iter=5):
        """
        Iterative sigma clipping that protects contiguous dips (real transits)
        from being removed as outliers. Only isolated single/double-point
        spikes (cosmic rays, detector glitches) are clipped — a clip is
        applied only if the point's immediate neighbours are NOT also
        deviant in the same direction (which would indicate a real transit).
        """
        mask = np.ones(len(flux), dtype=bool)
        for _ in range(max_iter):
            med = np.nanmedian(flux[mask])
            std = np.nanstd(flux[mask])
            deviant = np.abs(flux - med) > sigma * std

            # Protect contiguous deviant runs of length >= 3 — these are
            # transits/eclipses, not cosmic rays. Only clip isolated points.
            protected = deviant.copy()
            if deviant.any():
                # Find runs of consecutive True values
                idx = np.where(deviant)[0]
                run_start = 0
                for i in range(1, len(idx) + 1):
                    if i == len(idx) or idx[i] != idx[i-1] + 1:
                        run_len = i - run_start
                        if run_len >= 3:
                            # Real signal (transit) — un-protect (do not clip)
                            protected[idx[run_start:i]] = False
                        run_start = i

            new_mask = mask & ~protected
            if np.sum(new_mask) == np.sum(mask):
                break
            mask = new_mask
        return time[mask], flux[mask], flux_err[mask]

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE FOLD (static — used by classifier and visualizer)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def phase_fold(time, flux, period, t0,
                   n_bins_global=2001, n_bins_local=201):
        """
        Returns (phase_sorted, flux_sorted, global_view, local_view).
        global_view : 2001-point binned view, phase [-0.5, +0.5]
        local_view  : 201-point zoomed on transit, phase ±0.075
        """
        phase = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        idx   = np.argsort(phase)
        ps    = phase[idx]
        fs    = flux[idx]

        bins_g     = np.linspace(-0.5, 0.5, n_bins_global + 1)
        global_view = _bin_lc(ps, fs, bins_g)

        local_w    = 0.075
        bins_l     = np.linspace(-local_w, local_w, n_bins_local + 1)
        lmask      = np.abs(ps) <= local_w
        local_view = _bin_lc(ps[lmask], fs[lmask], bins_l) \
                     if lmask.sum() >= 5 else np.ones(n_bins_local)

        return ps, fs, global_view, local_view


def _bin_lc(phase, flux, bins):
    n = len(bins) - 1
    out = np.ones(n)
    for i in range(n):
        m = (phase >= bins[i]) & (phase < bins[i+1])
        if m.sum() > 0:
            out[i] = np.nanmean(flux[m])
    return out
