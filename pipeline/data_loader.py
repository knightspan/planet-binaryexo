"""
pipeline/data_loader.py
========================
Downloads REAL TESS 2-minute cadence light curves from NASA MAST archive
using the lightkurve library. Also provides a synthetic generator for
offline testing that uses physically realistic noise models.

REAL DATA SOURCES:
  - Primary:  SPOC 2-min cadence (NASA/MIT pipeline, highest quality)
  - Fallback: TESS-SPOC 10-min cadence
  - TPF:      Target Pixel Files for centroid motion analysis

SYNTHETIC DATA:
  - 27-day sector baseline (realistic TESS sector length)
  - Correlated 1/f + white noise (matches real TESS noise floor ~200-400 ppm)
  - Physically injected transit shapes (trapezoidal ingress/egress)
  - Secondary eclipses for EB simulation
  - Spot modulation for stellar variability simulation
"""

import numpy as np
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class TESSDataLoader:
    """Downloads real TESS light curves from MAST or generates synthetic ones."""

    def __init__(self, config: dict):
        self.config      = config
        self.last_sector = None
        self.cache_dir   = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────────────
    # REAL DATA: download from MAST
    # ─────────────────────────────────────────────────────────────────────────
    def download(self, tic_id: int, sector: int = None):
        """
        Download REAL TESS light curve for a TIC ID from NASA MAST.
        Returns a lightkurve LightCurve object or None.

        Priority order:
          1. SPOC 2-min (best quality, short cadence)
          2. TESS-SPOC 10-min (fallback)
        """
        try:
            import lightkurve as lk
        except ImportError:
            log.error("lightkurve not installed. Run: pip install lightkurve")
            return None

        for author, cadence in [("SPOC", "short"), ("TESS-SPOC", "long"),
                                  ("QLP", "long")]:
            try:
                search = lk.search_lightcurve(
                    f"TIC {tic_id}",
                    author   = author,
                    cadence  = cadence,
                    sector   = sector
                )
                if len(search) == 0:
                    continue

                log.info(f"  Found {len(search)} sector(s) via {author}/{cadence}")
                lc = search[0].download(
                    cache        = True,
                    download_dir = str(self.cache_dir)
                )
                self.last_sector = int(search[0].sector[0]) \
                    if hasattr(search[0], 'sector') and search[0].sector is not None else 0
                log.info(f"  Downloaded TIC {tic_id} sector {self.last_sector} "
                         f"via {author} — {len(lc.time)} cadences")
                return lc

            except Exception as e:
                log.debug(f"  {author} failed: {e}")
                continue

        log.warning(f"  No real TESS data found for TIC {tic_id}")
        return None

    def download_tpf(self, tic_id: int, sector: int = None):
        """Download Target Pixel File for centroid motion analysis."""
        try:
            import lightkurve as lk
            search = lk.search_targetpixelfile(
                f"TIC {tic_id}", author="SPOC", cadence="short", sector=sector)
            if len(search) == 0:
                return None
            return search[0].download(cache=True, download_dir=str(self.cache_dir))
        except Exception as e:
            log.debug(f"  TPF download failed: {e}")
            return None

    def download_sector_batch(self, sector: int, max_targets: int = 100):
        """
        Download all available SPOC targets in a TESS sector.
        Returns list of (tic_id, lc) tuples.
        This is what you use for the full 20,000–30,000 target sector run.
        """
        try:
            import lightkurve as lk
            log.info(f"Querying sector {sector} for all SPOC targets...")
            search = lk.search_lightcurve(
                "*", author="SPOC", cadence="short", sector=sector)
            log.info(f"  Found {len(search)} targets in sector {sector}")

            results = []
            for i, entry in enumerate(search[:max_targets]):
                try:
                    name   = str(entry.target_name).replace("TIC", "").strip()
                    tic_id = int(float(name))
                    lc     = entry.download(cache=True,
                                            download_dir=str(self.cache_dir))
                    results.append((tic_id, lc))
                    if (i + 1) % 10 == 0:
                        log.info(f"  Downloaded {i+1}/{min(max_targets, len(search))}")
                except Exception as e:
                    log.debug(f"  Skipping entry {i}: {e}")

            log.info(f"  Sector {sector}: {len(results)} LCs downloaded")
            return results

        except Exception as e:
            log.error(f"Sector batch download failed: {e}")
            return []

    # ─────────────────────────────────────────────────────────────────────────
    # SYNTHETIC DATA (offline testing / demo)
    # ─────────────────────────────────────────────────────────────────────────
    def get_synthetic_lc(self, seed: int, signal_type: str = 'planet_transit'):
        """
        Generate a physically realistic synthetic TESS light curve.

        signal_type options:
          'planet_transit'   — shallow flat-bottomed dip, depth 0.05–2%
          'eclipsing_binary' — deep V-shaped eclipse + secondary eclipse
          'blend'            — diluted EB eclipse from background star
          'noise'            — pure correlated noise, no signal
          'stellar_variability' — sinusoidal spot modulation

        Returns dict with keys: time, flux, flux_err, tic_id, sector, signal_type
        """
        rng = np.random.default_rng(seed)

        # 27-day sector at 2-min cadence = 19440 points
        n      = 19440
        time   = np.linspace(0.0, 27.0, n)

        # ── Realistic noise model ────────────────────────────────────────────
        # White noise: 200–400 ppm typical for bright TESS targets
        wn_amp    = rng.uniform(200e-6, 400e-6)
        white     = rng.normal(0, wn_amp, n)

        # Correlated 1/f noise: instrumental systematics
        red       = np.cumsum(rng.normal(0, wn_amp * 0.3, n))
        red      -= np.mean(red)
        red      /= (np.std(red) + 1e-12)
        red      *= wn_amp * 0.5

        # Slow trend: thermal drift
        trend     = 0.0003 * np.sin(2 * np.pi * time / 13.5 + rng.uniform(0, 2*np.pi))

        flux      = 1.0 + white + red + trend
        flux_err  = np.abs(white) + wn_amp * 0.5

        # ── Inject astrophysical signal ──────────────────────────────────────
        if signal_type == 'planet_transit':
            period   = rng.uniform(1.5, 9.0)
            # Realistic detectable TESS planet depths: 300 ppm (super-Earth)
            # to 2% (hot Jupiter). Below ~300ppm is below typical single-sector
            # detection threshold given ~300-400 ppm noise floor — matches
            # real TESS planet discovery statistics.
            depth    = rng.uniform(0.0008, 0.02)   # 800 ppm – 2%
            duration = rng.uniform(0.06, 0.14)     # days
            t0       = rng.uniform(0.2, period)
            flux     = self._inject_transit(time, flux, period, depth,
                                             duration, t0, rng, shape='flat')

        elif signal_type == 'eclipsing_binary':
            period   = rng.uniform(1.0, 5.0)
            depth    = rng.uniform(0.05, 0.40)     # 5–40% deep
            duration = rng.uniform(0.04, 0.12)
            t0       = rng.uniform(0.2, period)
            flux     = self._inject_transit(time, flux, period, depth,
                                             duration, t0, rng, shape='vshaped')
            # Secondary eclipse at phase 0.5
            sec_depth = depth * rng.uniform(0.3, 0.8)
            flux      = self._inject_transit(time, flux, period, sec_depth,
                                              duration, t0 + period * 0.5, rng,
                                              shape='vshaped')

        elif signal_type == 'blend':
            # Background EB diluted by target star flux
            period     = rng.uniform(1.0, 5.0)
            true_depth = rng.uniform(0.05, 0.20)
            dilution   = rng.uniform(0.05, 0.15)   # fraction from BG star
            depth      = true_depth * dilution      # apparent depth much smaller
            duration   = rng.uniform(0.04, 0.10)
            t0         = rng.uniform(0.2, period)
            flux       = self._inject_transit(time, flux, period, depth,
                                               duration, t0, rng, shape='vshaped')
            # Slight centroid shift signature in noise (for centroid test)
            flux      += 0.0001 * np.sin(2 * np.pi * time / period)

        elif signal_type == 'stellar_variability':
            # Quasi-sinusoidal spot modulation
            prot     = rng.uniform(5.0, 20.0)
            amp      = rng.uniform(0.003, 0.015)
            flux    += amp * np.sin(2 * np.pi * time / prot + rng.uniform(0, 2*np.pi))
            # Spot evolution: amplitude drift
            flux    += amp * 0.3 * np.sin(2 * np.pi * time / (prot * 2.3))

        # signal_type == 'noise': no injection needed

        # Momentum dumps: brief gaps at ~3-day intervals (real TESS systematics)
        for dump_time in np.arange(3, 27, 3.5):
            mask = np.abs(time - dump_time) < 0.02
            flux[mask] = np.nan

        # Remove NaN for consistency
        valid     = np.isfinite(flux)
        time      = time[valid]
        flux      = flux[valid]
        flux_err  = flux_err[valid]

        return {
            'time':         time,
            'flux':         flux,
            'flux_err':     flux_err,
            'tic_id':       seed,
            'sector':       99,
            'synthetic':    True,
            'signal_type':  signal_type,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────
    def _inject_transit(self, time, flux, period, depth, duration,
                         t0, rng, shape='flat'):
        """Inject transit/eclipse with realistic ingress/egress shape."""
        flux = flux.copy()
        half = duration / 2.0
        ingress_frac = 0.15 if shape == 'flat' else 0.45

        t = t0
        while t < time[-1]:
            if t >= time[0] - half:
                dt = np.abs(time - t)
                ingress_w = half * ingress_frac
                full_w    = half * (1 - ingress_frac)

                # Full transit floor
                in_full   = dt <= full_w
                flux[in_full] -= depth

                # Ingress / egress ramp
                in_ramp   = (dt > full_w) & (dt <= half)
                ramp      = (half - dt[in_ramp]) / max(ingress_w, 1e-9)
                flux[in_ramp] -= depth * np.clip(ramp, 0, 1)

            t += period

        # Also walk backwards
        t = t0 - period
        while t >= time[0] - half:
            dt = np.abs(time - t)
            ingress_w = half * ingress_frac
            full_w    = half * (1 - ingress_frac)
            in_full   = dt <= full_w
            flux[in_full] -= depth
            in_ramp   = (dt > full_w) & (dt <= half)
            ramp      = (half - dt[in_ramp]) / max(ingress_w, 1e-9)
            flux[in_ramp] -= depth * np.clip(ramp, 0, 1)
            t -= period

        return flux
