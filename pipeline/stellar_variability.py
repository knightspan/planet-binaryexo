"""
pipeline/stellar_variability.py
================================
Pre-screen for stellar rotational variability (starspots) BEFORE transit
search. Runs a Lomb-Scargle periodogram on the RAW (un-detrended) light
curve and distinguishes spot modulation from transit signals by shape.

Key differences between starspots and transits:
  Starspots  → broad sinusoidal wave, duty cycle > 25%, amplitude drifts
  Transits   → narrow flat-bottomed dip, duty cycle < 10%, stable depth

Thresholds (conservative — prefer false negatives over false positives):
  fap_threshold       = 1e-3   (FAP of LS peak)
  amp_ratio_threshold = 5.0    (peak-to-peak / noise floor)
  duty_cycle_threshold= 0.25   (fraction of phase below midpoint)
  All three must fire together to route to stellar_variability class.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class StellarVariabilityScreener:

    def __init__(self, config: dict):
        self.fap_threshold        = config.get('rotation_fap_threshold',    1e-3)
        self.amp_ratio_threshold  = config.get('spot_amp_ratio_threshold',  5.0)
        self.duty_cycle_threshold = config.get('spot_duty_cycle_threshold', 0.25)
        self.period_min           = config.get('rotation_period_min',       0.1)
        self.period_max           = config.get('rotation_period_max',      30.0)

    def screen(self, time, flux_raw, flux_err):
        """
        Parameters
        ----------
        time, flux_raw, flux_err : array-like
            RAW (un-detrended) light curve.

        Returns
        -------
        dict:
            rotation_period    : float or None
            rotation_fap       : float or None
            amplitude_ratio    : float or None
            duty_cycle         : float or None
            is_spot_dominated  : bool
            recommendation     : 'proceed_to_tls' | 'flag_for_review' |
                                  'route_to_variability_class'
        """
        time     = np.asarray(time,     dtype=np.float64)
        flux_raw = np.asarray(flux_raw, dtype=np.float64)
        flux_err = np.asarray(flux_err, dtype=np.float64)

        # Need at least astropy and enough points
        if len(time) < 100:
            return self._null('insufficient_data')

        try:
            from astropy.timeseries import LombScargle
        except ImportError:
            return self._null('astropy_missing')

        try:
            ls = LombScargle(time, flux_raw, flux_err)
            freq, power = ls.autopower(
                minimum_frequency = 1.0 / self.period_max,
                maximum_frequency = 1.0 / self.period_min,
                samples_per_peak  = 10,
            )
            best_idx    = int(np.argmax(power))
            best_period = float(1.0 / freq[best_idx])
            best_power  = float(power[best_idx])
            fap         = float(ls.false_alarm_probability(best_power,
                                                            method='baluev'))

            # Phase-fold at candidate rotation period
            phase = np.mod(time, best_period) / best_period
            order = np.argsort(phase)

            n_bins    = 20
            bm, _     = np.histogram(phase[order], bins=n_bins,
                                      range=(0, 1), weights=flux_raw[order])
            bc, _     = np.histogram(phase[order], bins=n_bins, range=(0, 1))
            bc        = np.where(bc == 0, 1, bc)
            binned    = bm / bc

            amplitude     = float(np.nanmax(binned) - np.nanmin(binned))
            noise_floor   = float(np.nanmedian(np.abs(flux_err))) + 1e-8
            amp_ratio     = amplitude / noise_floor

            mid           = (np.nanmax(binned) + np.nanmin(binned)) / 2.0
            duty_cycle    = float(np.mean(binned < mid))

            is_spot = (fap        < self.fap_threshold and
                       amp_ratio  > self.amp_ratio_threshold and
                       duty_cycle > self.duty_cycle_threshold)

            if is_spot:
                rec = 'route_to_variability_class'
            elif fap < self.fap_threshold and duty_cycle > 0.15:
                rec = 'flag_for_review'
            else:
                rec = 'proceed_to_tls'

            log.info(f"  Variability screen: P={best_period:.2f}d  "
                     f"FAP={fap:.2e}  amp_ratio={amp_ratio:.1f}  "
                     f"duty={duty_cycle:.2f}  → {rec}")

            return {
                'rotation_period':   best_period,
                'rotation_power':    best_power,
                'rotation_fap':      fap,
                'amplitude_ratio':   amp_ratio,
                'duty_cycle':        duty_cycle,
                'is_spot_dominated': is_spot,
                'recommendation':    rec,
            }

        except Exception as e:
            log.debug(f"  Variability screen error: {e}")
            return self._null(str(e))

    @staticmethod
    def _null(reason=''):
        return {
            'rotation_period':   None,
            'rotation_power':    None,
            'rotation_fap':      None,
            'amplitude_ratio':   None,
            'duty_cycle':        None,
            'is_spot_dominated': False,
            'recommendation':    'proceed_to_tls',
            'skip_reason':       reason,
        }
