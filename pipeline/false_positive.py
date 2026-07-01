"""
pipeline/false_positive.py
===========================
Five independent false-positive vetting tests, each targeting a specific
physical cause of transit-like signals that are NOT planets.

Test 1 — Odd/Even depth ratio
  Eclipsing binaries produce alternating deep/shallow eclipses because the
  two stars eclipse each other alternately. True planet transits show equal
  depth at every epoch. Ratio |odd_mean / even_mean| deviating from 1 by
  more than 0.15 flags an EB.

Test 2 — Secondary eclipse search
  Eclipsing binaries show a secondary eclipse at orbital phase 0.5
  (the other star passing behind). Planets don't. Searched in a window
  ±1.5× transit duration around phase 0.5.

Test 3 — Centroid motion (requires TPF download)
  If the photometric centroid shifts during transit, the signal comes from
  a background star, not the target. TESS pixels are 21 arcsec so shifts
  > 0.5 px are significant.

Test 4 — V-shape metric
  Real planet transits have a flat bottom. Eclipsing binaries and grazing
  transits are V-shaped. Metric = (bottom-third depth) / (total depth).
  Close to 1 = flat-bottomed planet; close to 0 = V-shaped EB.

Test 5 — Transit depth variability
  Real planets produce consistent transit depth every epoch. Blends and
  EBs often show variable depth. std/mean > 0.3 is suspicious.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class FalsePositiveVetter:

    def __init__(self, config: dict):
        self.config = config

    def vet(self, time, flux, flux_err, tls_result,
            run_centroid: bool = True, tic_id: int = None):
        """
        Run all five tests. Returns dict of flag values.
        run_centroid requires internet (TPF download).
        """
        flags = {}

        # 1. Odd/Even
        flags['odd_even_ratio']    = self._odd_even(time, flux, tls_result)

        # 2. Secondary eclipse
        sec_d, sec_snr             = self._secondary(time, flux, tls_result)
        flags['secondary_depth']   = sec_d
        flags['secondary_snr']     = sec_snr

        # 3. V-shape
        flags['v_shape_metric']    = self._v_shape(time, flux, tls_result)

        # 4. Depth variability
        dm, ds                     = self._depth_var(time, flux, tls_result)
        flags['depth_mean']        = dm
        flags['depth_std']         = ds

        # 5. Transit count
        flags['n_transits']        = int(tls_result.transit_count or 0)
        flags['tls_odd_even']      = float(tls_result.odd_even_mismatch or 1.0)

        # 6. Centroid (optional — requires TPF)
        if run_centroid and tic_id is not None:
            flags['centroid_shift'] = self._centroid(time, tls_result, tic_id)
        else:
            flags['centroid_shift'] = 0.0

        # Composite FP score 0–5
        flags['fp_indicators'] = self._fp_score(flags, tls_result)

        log.info(f"  Vetting: odd/even={flags['odd_even_ratio']:.3f}  "
                 f"sec_depth={flags['secondary_depth']:.5f}  "
                 f"v_shape={flags['v_shape_metric']:.3f}  "
                 f"centroid={flags['centroid_shift']:.3f}px  "
                 f"fp_score={flags['fp_indicators']}/5")
        return flags

    # ── Test 1: Odd/Even ─────────────────────────────────────────────────────
    def _odd_even(self, time, flux, r):
        period = float(r.period)
        t0     = float(r.T0)
        dur    = float(r.duration)

        odd, even = [], []
        t = t0; epoch = 0
        while t < time[-1]:
            if t >= time[0]:
                mask = np.abs(time - t) < dur * 1.5
                if mask.sum() >= 3:
                    d = 1.0 - np.nanmin(flux[mask])
                    if d > 0:
                        (even if epoch % 2 == 0 else odd).append(d)
            t += period; epoch += 1
        # Walk backward
        t = t0 - period; epoch = -1
        while t >= time[0]:
            mask = np.abs(time - t) < dur * 1.5
            if mask.sum() >= 3:
                d = 1.0 - np.nanmin(flux[mask])
                if d > 0:
                    (even if abs(epoch) % 2 == 0 else odd).append(d)
            t -= period; epoch -= 1

        if odd and even:
            return float(np.mean(odd) / (np.mean(even) + 1e-9))
        return 1.0

    # ── Test 2: Secondary eclipse ─────────────────────────────────────────────
    def _secondary(self, time, flux, r):
        period = float(r.period)
        t0     = float(r.T0)
        dur    = float(r.duration)

        t_sec  = t0 + period / 2.0
        phase  = ((time - t_sec) % period) / period
        phase[phase > 0.5] -= 1.0
        window = (dur / period) * 1.5
        in_s   = np.abs(phase) < window
        out_s  = np.abs(phase) > window * 3

        if in_s.sum() < 3 or out_s.sum() < 5:
            return 0.0, 0.0

        depth  = max(float(np.nanmedian(flux[out_s]) - np.nanmin(flux[in_s])), 0.0)
        noise  = np.nanstd(flux[out_s]) / np.sqrt(in_s.sum())
        snr    = depth / max(noise, 1e-12)
        return depth, float(snr)

    # ── Test 3: V-shape ───────────────────────────────────────────────────────
    def _v_shape(self, time, flux, r):
        period = float(r.period)
        t0     = float(r.T0)
        dur    = float(r.duration)
        phase  = ((time - t0) % period) / period
        phase[phase > 0.5] -= 1.0
        half   = dur / period / 2.0

        total  = np.abs(phase) < half
        bottom = np.abs(phase) < half / 3.0

        if total.sum() < 5:
            return 0.5

        d_total  = 1.0 - np.nanmin(flux[total])
        d_bottom = 1.0 - np.nanmin(flux[bottom]) if bottom.sum() >= 2 else d_total
        if d_total < 1e-6:
            return 0.5
        return float(np.clip(d_bottom / (d_total + 1e-12), 0, 2))

    # ── Test 4: Depth variability ─────────────────────────────────────────────
    def _depth_var(self, time, flux, r):
        period = float(r.period)
        t0     = float(r.T0)
        dur    = float(r.duration)

        depths = []
        t = t0
        while t < time[-1]:
            if t >= time[0]:
                mask = np.abs(time - t) < dur
                if mask.sum() >= 5:
                    d = 1.0 - np.nanmin(flux[mask])
                    if d > 0:
                        depths.append(d)
            t += period

        if len(depths) >= 2:
            return float(np.mean(depths)), float(np.std(depths))
        return float(r.depth), 0.0

    # ── Test 5: Centroid ──────────────────────────────────────────────────────
    def _centroid(self, time, r, tic_id):
        try:
            from pipeline.data_loader import TESSDataLoader
            loader = TESSDataLoader(self.config)
            tpf    = loader.download_tpf(tic_id)
            if tpf is None:
                return 0.0

            col, row = tpf.estimate_centroids()
            col_v    = np.array(col.value, dtype=np.float64)
            row_v    = np.array(row.value, dtype=np.float64)
            tpf_time = np.array(tpf.time.value, dtype=np.float64)

            period   = float(r.period)
            t0       = float(r.T0)
            dur      = float(r.duration)
            phase    = ((tpf_time - t0) % period) / period
            phase[phase > 0.5] -= 1.0
            half     = dur / period / 2.0 * 1.2
            in_tr    = np.abs(phase) < half
            out_tr   = np.abs(phase) > half * 3

            if in_tr.sum() < 3 or out_tr.sum() < 5:
                return 0.0

            dc = np.nanmean(col_v[in_tr]) - np.nanmean(col_v[out_tr])
            dr = np.nanmean(row_v[in_tr]) - np.nanmean(row_v[out_tr])
            shift = float(np.sqrt(dc**2 + dr**2))
            log.info(f"  Centroid shift: {shift:.3f} px")
            return shift

        except Exception as e:
            log.debug(f"  Centroid test skipped: {e}")
            return 0.0

    # ── Composite FP score ────────────────────────────────────────────────────
    def _fp_score(self, flags, r):
        score = 0
        oe = flags.get('odd_even_ratio', 1.0)
        if abs(oe - 1.0) > 0.15:                        score += 1
        if flags.get('secondary_snr', 0) > 3:           score += 1
        if flags.get('centroid_shift', 0) > 0.5:        score += 1
        if float(r.depth) > 0.01:                       score += 1
        if flags.get('v_shape_metric', 1) < 0.4:        score += 1
        return score
