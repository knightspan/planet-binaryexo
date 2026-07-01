"""
pipeline/multiplanet.py
========================
Iterative multi-planet transit search.

After the primary TLS detection, mask the in-transit points of the found
signal and re-run TLS on the residual. Repeat until:
  - SDE of the next signal falls below the threshold, OR
  - max_planets limit is reached, OR
  - the next period is too close to an already-found period (< 10% difference)

This is the standard approach for multi-planet systems (e.g. Kepler-90, TRAPPIST-1)
where a single TLS pass only finds the strongest signal.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


def iterative_tls_search(time, flux, flux_err, detector,
                          max_planets=3, sde_threshold=7.0):
    """
    Parameters
    ----------
    time, flux, flux_err : ndarray   detrended light curve
    detector             : TransitDetector instance
    max_planets          : int       maximum signals to extract
    sde_threshold        : float     stop when best SDE < this

    Returns
    -------
    list of TLSResult objects (first = primary detection).
    May be empty if no primary signal exceeds threshold.
    """
    detections = []

    # Primary search on unmasked curve
    primary = detector.search(time, flux, flux_err)
    if primary is None or primary.SDE < sde_threshold:
        return detections

    detections.append(primary)
    flux_work = flux.copy()

    for planet_n in range(1, max_planets):
        prev      = detections[-1]
        flux_work = _mask_transits(time, flux_work,
                                    prev.period, prev.T0, prev.duration)

        try:
            residual = detector.search(time, flux_work, flux_err)
        except Exception as e:
            log.debug(f"  Multi-planet iter {planet_n} failed: {e}")
            break

        if residual is None or residual.SDE < sde_threshold:
            break

        # Check period is genuinely different (> 10% from all found)
        is_new = all(
            abs(residual.period - d.period) / max(d.period, 1e-3) > 0.10
            for d in detections
        )
        if not is_new:
            log.debug(f"  Multi-planet: period {residual.period:.4f}d too "
                      f"close to existing signal — stopping")
            break

        log.info(f"  Multi-planet signal {planet_n + 1}: "
                 f"P={residual.period:.4f}d  "
                 f"depth={residual.depth*100:.4f}%  "
                 f"SDE={residual.SDE:.2f}")
        detections.append(residual)

    return detections


def _mask_transits(time, flux, period, t0, duration, margin=1.5):
    """Replace in-transit points with local out-of-transit median."""
    flux_m = flux.copy()
    half   = (duration * margin) / 2.0

    for direction in [+1, -1]:
        t = t0
        while True:
            if direction == +1:
                if t > time[-1]: break
            else:
                if t < time[0]:  break
            if time[0] - half <= t <= time[-1] + half:
                mask = np.abs(time - t) < half
                if mask.sum() > 0:
                    out  = ~mask
                    fill = np.nanmedian(flux_m[out]) if out.sum() > 0 else 1.0
                    flux_m[mask] = fill
            t += direction * period

    return flux_m
