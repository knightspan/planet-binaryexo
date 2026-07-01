"""
pipeline/blend_crosscheck.py
=============================
Independent catalog-based corroboration for blend false positives.
A single centroid measurement on noisy TESS pixels (21 arcsec/pixel)
is not always reliable. This module adds two independent catalog signals:

  1. TIC contamination ratio — fraction of aperture flux from other sources,
     already computed by the TIC-8 catalog for every TESS target.
  2. Gaia DR3 neighbour search — checks if any nearby bright source could
     produce the observed (diluted) transit depth.

A "blend" call requires:
  centroid_shift >= 0.5 px   (direct evidence)
  OR  contamination > 10% AND Gaia neighbour within aperture

This prevents a single noisy measurement from driving a false "blend" verdict.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)

TESS_PIXEL_ARCSEC = 21.0


class BlendCrossChecker:

    def __init__(self, config: dict):
        self.contam_thresh   = config.get('contamination_ratio_threshold', 0.10)
        self.search_radius   = config.get('blend_search_radius_px',        2.0)
        self.max_mag_diff    = config.get('blend_max_neighbor_mag_diff',    4.0)

    def check(self, tic_id, observed_depth_ppm,
              centroid_shift_px=None,
              tic_contamination_ratio=None,
              gaia_neighbors=None):
        """
        Parameters
        ----------
        tic_id                  : int
        observed_depth_ppm      : float   transit depth in ppm
        centroid_shift_px       : float or None   from FP vetter centroid test
        tic_contamination_ratio : float or None   from TIC catalog 'contratio'
        gaia_neighbors          : list[dict] or None
            Each dict: {'separation_arcsec': float, 'g_mag_diff': float}

        Returns
        -------
        dict:
            blend_likely        : bool
            evidence            : list[str]
            n_independent_flags : int (0-3)
            contamination_flag  : bool
            neighbor_flag       : bool
            centroid_flag       : bool
        """
        evidence           = []
        contamination_flag = False
        neighbor_flag      = False
        centroid_flag      = False

        # Test 1: TIC contamination ratio
        if tic_contamination_ratio is not None:
            contamination_flag = tic_contamination_ratio > self.contam_thresh
            if contamination_flag:
                evidence.append(
                    f"TIC contamination ratio {tic_contamination_ratio:.3f} "
                    f"> threshold {self.contam_thresh:.3f}")

        # Test 2: Gaia neighbour proximity
        if gaia_neighbors:
            radius_arcsec = self.search_radius * TESS_PIXEL_ARCSEC
            close_bright  = [
                n for n in gaia_neighbors
                if n.get('separation_arcsec', 999) <= radius_arcsec
                and n.get('g_mag_diff', 99) <= self.max_mag_diff
            ]
            neighbor_flag = len(close_bright) > 0
            if neighbor_flag:
                best = min(close_bright, key=lambda n: n['separation_arcsec'])
                evidence.append(
                    f"Gaia neighbour {best['separation_arcsec']:.1f}\" away, "
                    f"Δmag={best['g_mag_diff']:.2f} (inside aperture)")

        # Test 3: Centroid shift
        if centroid_shift_px is not None:
            centroid_flag = centroid_shift_px >= 0.5
            if centroid_flag:
                evidence.append(
                    f"Centroid shift {centroid_shift_px:.2f} px during transit")

        n_flags     = sum([contamination_flag, neighbor_flag, centroid_flag])
        blend_likely = centroid_flag or (contamination_flag and neighbor_flag)

        if not evidence:
            evidence.append("No blend indicators triggered")

        return {
            'blend_likely':        blend_likely,
            'evidence':            evidence,
            'n_independent_flags': n_flags,
            'contamination_flag':  contamination_flag,
            'neighbor_flag':       neighbor_flag,
            'centroid_flag':       centroid_flag,
        }
