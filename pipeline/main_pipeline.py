"""
pipeline/main_pipeline.py
==========================
Master orchestrator — wires all 10 pipeline stages together.

Stage order:
  1.  Download real TESS LC from MAST (or use synthetic dict)
  2.  Stellar variability pre-screen (Lomb-Scargle starspot routing)
  3.  Wotan biweight detrending + sigma clipping
  4.  TLS transit search + iterative multi-planet search
  5.  SNR computation (folded + TLS native + SDE)
  6.  Bootstrap permutation False Alarm Probability
  7.  Five-test false-positive vetting (odd/even, secondary, centroid, V-shape, depth-var)
  8.  Independent blend cross-check (TIC contamination + Gaia neighbours)
  9.  Dual-view CNN + rule ensemble classification + calibrated confidence
  10. batman MAP → emcee MCMC parameter fitting (planet candidates only)

Then: 6-panel diagnostic plot + MCMC corner plot + auto-report.
"""

import numpy as np
import pandas as pd
import warnings
import logging
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
log = logging.getLogger(__name__)

from pipeline.data_loader          import TESSDataLoader
from pipeline.preprocessor         import LightCurvePreprocessor
from pipeline.detector             import TransitDetector
from pipeline.classifier           import ExoplanetClassifier, CLASSES
from pipeline.fitter               import TransitParameterFitter
from pipeline.false_positive       import FalsePositiveVetter
from pipeline.snr_calculator       import SNRCalculator
from pipeline.stellar_variability  import StellarVariabilityScreener
from pipeline.blend_crosscheck     import BlendCrossChecker
from pipeline.significance         import SignificanceEstimator
from pipeline.multiplanet          import iterative_tls_search
from utils.report_generator        import ReportGenerator
from utils.visualizer              import PipelineVisualizer


# ─────────────────────────────────────────────────────────────────────────────
# RESULT CONTAINER
# ─────────────────────────────────────────────────────────────────────────────
class PipelineResult:
    """Holds all outputs for one star."""

    def __init__(self, tic_id: int):
        self.tic_id             = tic_id
        self.time               = None
        self.flux               = None
        self.flux_err           = None
        self.sector             = None
        self.classification     = None
        self.confidence         = None
        self.class_probs        = None
        self.tls_result         = None
        self.extra_planets      = []
        self.transit_params     = None
        self.mcmc_samples       = None
        self.snr                = None
        self.fap                = None
        self.fap_details        = None
        self.vetting_flags      = {}
        self.centroid_shift     = None
        self.odd_even_ratio     = None
        self.secondary_depth    = None
        self.variability_screen = None
        self.blend_crosscheck   = None
        self.plot_path          = None
        self.status             = 'pending'
        self.error              = None

    def to_dict(self) -> dict:
        d = {
            'tic_id':         self.tic_id,
            'classification': self.classification,
            'confidence':     round(float(self.confidence), 4) if self.confidence is not None else None,
            'snr':            round(float(self.snr), 2)        if self.snr        is not None else None,
            'fap':            float(self.fap)                  if self.fap        is not None else None,
            'status':         self.status,
            'n_planets_found': 1 + len(self.extra_planets or []),
        }
        if self.class_probs:
            d['class_probabilities'] = {
                k: round(float(v), 4) for k, v in self.class_probs.items()}
        if self.tls_result is not None:
            r = self.tls_result
            d.update({
                'period_days':    round(float(r.period), 6),
                'period_err':     round(float(r.period_uncertainty), 6),
                't0_btjd':        round(float(r.T0), 6),
                'transit_depth':  round(float(r.depth), 6),
                'depth_pct':      round(float(r.depth * 100), 4),
                'duration_hours': round(float(r.duration * 24), 4),
                'sde':            round(float(r.SDE), 2),
                'transit_count':  int(r.transit_count or 0),
            })
        if self.transit_params:
            tp = self.transit_params
            d.update({
                'fit_period':          round(tp['period'], 6),
                'fit_period_err':      round(tp['period_err'], 6),
                'fit_depth_pct':       round(tp['transit_depth'] * 100, 4),
                'fit_depth_err_pct':   round(tp['transit_depth_err'] * 100, 4),
                'fit_duration_h':      round(tp['transit_duration'] * 24, 3),
                'fit_rp_rs':           round(tp['rp_rs'], 5),
                'fit_rp_rs_err':       round(tp['rp_rs_err'], 5),
                'fit_a_rs':            round(tp['a_rs'], 2),
                'fit_inclination':     round(tp['inclination'], 2),
                'fit_mcmc_used':       tp['mcmc_used'],
            })
        if self.variability_screen:
            vs = self.variability_screen
            d['rotation_period'] = vs.get('rotation_period')
            d['rotation_fap']    = vs.get('rotation_fap')
        if self.vetting_flags:
            for k in ['odd_even_ratio', 'secondary_depth', 'secondary_snr',
                      'centroid_shift', 'v_shape_metric', 'fp_indicators']:
                if k in self.vetting_flags:
                    d[f'vet_{k}'] = self.vetting_flags[k]
        return d


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class ExoplanetPipeline:

    def __init__(self, config: dict = None):
        self.config     = config or _default_config()
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader     = TESSDataLoader(self.config)
        self.pre        = LightCurvePreprocessor(self.config)
        self.det        = TransitDetector(self.config)
        self.clf        = ExoplanetClassifier(self.config)
        self.fitter     = TransitParameterFitter(self.config)
        self.vetter     = FalsePositiveVetter(self.config)
        self.snr_c      = SNRCalculator()
        self.var_s      = StellarVariabilityScreener(self.config)
        self.blend_c    = BlendCrossChecker(self.config)
        self.sig        = SignificanceEstimator(self.config)
        self.viz        = PipelineVisualizer(self.config)
        self.rep        = ReportGenerator(self.config)
        self.results    = []

    # ─────────────────────────────────────────────────────────────────────────
    def analyse_star(self, tic_id: int, sector: int = None,
                     lc_input=None) -> PipelineResult:
        """
        Analyse one star end-to-end.
        lc_input: pre-loaded LC dict (synthetic) or None (triggers download).
        """
        result = PipelineResult(tic_id)
        log.info(f"{'='*60}")
        log.info(f"  TIC {tic_id}")

        try:
            # ── 1. Data acquisition ─────────────────────────────────────────
            if lc_input is None:
                log.info("Stage 1/10 — Downloading TESS LC from MAST")
                lc = self.loader.download(tic_id, sector=sector)
                if lc is None:
                    result.status = 'no_data'; return result
                result.sector = getattr(self.loader, 'last_sector', None)
            else:
                lc = lc_input
                result.sector = lc.get('sector', 99) if isinstance(lc, dict) else None

            # ── 2. Stellar variability pre-screen ───────────────────────────
            if self.config.get('run_variability_screen', True):
                log.info("Stage 2/10 — Stellar variability pre-screen")
                try:
                    if isinstance(lc, dict):
                        _t = lc['time']; _f = lc['flux']; _e = lc['flux_err']
                    else:
                        _t = lc.time.value
                        _f = lc.flux.value
                        _e = lc.flux_err.value if hasattr(lc, 'flux_err') else np.ones_like(_f) * 1e-4
                    var_r = self.var_s.screen(_t, _f, _e)
                    result.variability_screen = var_r
                    if var_r['recommendation'] == 'route_to_variability_class':
                        log.info(f"  → stellar_variability "
                                 f"(P={var_r.get('rotation_period', 0):.2f}d)")
                        result.classification = 'stellar_variability'
                        result.confidence     = 0.90
                        result.class_probs    = {c: 0.0 for c in CLASSES}
                        result.snr = 0.0
                        result.status = 'done'
                        return result
                except Exception as e:
                    log.debug(f"  Variability screen error (non-fatal): {e}")

            # ── 3. Preprocess ───────────────────────────────────────────────
            log.info("Stage 3/10 — Wotan biweight detrend")
            time, flux, flux_err = self.pre.process(lc)
            result.time     = time
            result.flux     = flux
            result.flux_err = flux_err

            # ── 4. TLS + multi-planet ───────────────────────────────────────
            log.info("Stage 4/10 — TLS transit search")
            tls = self.det.search(time, flux, flux_err)
            if tls is None or tls.SDE < 6.0:
                log.info(f"  SDE={getattr(tls,'SDE',0):.2f} < 6.0 — no signal")
                result.status = 'no_signal'; return result

            result.tls_result = tls
            log.info(f"  P={tls.period:.4f}d  depth={tls.depth*100:.4f}%  SDE={tls.SDE:.2f}")

            if self.config.get('run_multiplanet_search', True):
                planets = iterative_tls_search(
                    time, flux, flux_err, self.det,
                    max_planets=self.config.get('max_planets_per_target', 3),
                    sde_threshold=self.config.get('multiplanet_sde_threshold', 7.0))
                result.extra_planets = planets[1:]
                if result.extra_planets:
                    log.info(f"  Multi-planet: {len(result.extra_planets)} additional signal(s)")

            # ── 5. SNR ──────────────────────────────────────────────────────
            log.info("Stage 5/10 — SNR computation")
            result.snr = self.snr_c.compute(time, flux, flux_err, tls)
            if result.snr < self.config.get('min_snr', 7.0):
                log.info(f"  SNR={result.snr:.2f} below threshold — skipping")
                result.status = 'low_snr'; return result
            log.info(f"  SNR={result.snr:.2f}  [{self.snr_c.significance_statement(result.snr)}]")

            # ── 6. Bootstrap FAP ────────────────────────────────────────────
            if self.config.get('run_fap_bootstrap', True):
                log.info("Stage 6/10 — Bootstrap FAP")
                try:
                    fap_res       = self.sig.bootstrap_fap(
                        time, flux, flux_err, self.det, tls.SDE,
                        n_bootstrap=self.config.get('fap_n_bootstrap', 100))
                    result.fap         = fap_res['fap']
                    result.fap_details = fap_res
                    log.info(f"  FAP={result.fap:.2e}  "
                             f"[{SignificanceEstimator.significance_statement(result.fap)}]")
                except Exception as e:
                    log.warning(f"  FAP bootstrap failed (non-fatal): {e}")

            # ── 7. FP vetting ────────────────────────────────────────────────
            log.info("Stage 7/10 — False-positive vetting")
            flags = self.vetter.vet(
                time, flux, flux_err, tls,
                run_centroid=self.config.get('run_centroid', True),
                tic_id=tic_id)
            result.vetting_flags   = flags
            result.odd_even_ratio  = flags.get('odd_even_ratio', 1.0)
            result.secondary_depth = flags.get('secondary_depth', 0.0)
            result.centroid_shift  = flags.get('centroid_shift', 0.0)

            # ── 8. Blend cross-check ─────────────────────────────────────────
            if self.config.get('run_blend_crosscheck', True):
                log.info("Stage 8/10 — Blend cross-check")
                try:
                    bc = self.blend_c.check(
                        tic_id=tic_id,
                        observed_depth_ppm=tls.depth * 1e6,
                        centroid_shift_px=flags.get('centroid_shift', 0.0))
                    flags['blend_crosscheck']  = bc
                    result.blend_crosscheck    = bc
                except Exception as e:
                    log.debug(f"  Blend cross-check skipped: {e}")

            # ── 9. Classify ──────────────────────────────────────────────────
            log.info("Stage 9/10 — Classification")
            label, conf, probs = self.clf.predict(time, flux, tls, flags)
            result.classification = label
            result.confidence     = conf
            result.class_probs    = {CLASSES[i]: float(probs[i]) for i in range(4)}
            log.info(f"  → {label}  confidence={conf*100:.1f}%")

            # ── 10. Parameter fitting ─────────────────────────────────────────
            if (label == 'planet_transit' and
                    conf >= self.config.get('confidence_threshold', 0.70)):
                log.info("Stage 10/10 — batman MAP + emcee MCMC")
                tp, samples = self.fitter.fit(
                    time, flux, flux_err, tls,
                    run_mcmc=self.config.get('run_mcmc', False))
                result.transit_params = tp
                result.mcmc_samples   = samples
                if tp:
                    log.info(f"  Period:   {tp['period']:.5f} ± {tp['period_err']:.5f} d")
                    log.info(f"  Depth:    {tp['transit_depth']*100:.4f} ± "
                             f"{tp['transit_depth_err']*100:.4f} %")
                    log.info(f"  Duration: {tp['transit_duration']*24:.3f} ± "
                             f"{tp['transit_duration_err']*24:.3f} h")
                    log.info(f"  Rp/Rs:    {tp['rp_rs']:.5f} ± {tp['rp_rs_err']:.5f}"
                             f"  {'[MCMC]' if tp['mcmc_used'] else '[MAP]'}")
            else:
                log.info("Stage 10/10 — Skipping MCMC (non-planet or low confidence)")

            result.status = 'done'

        except Exception as e:
            log.error(f"Pipeline error TIC {tic_id}: {e}", exc_info=True)
            result.status = 'error'
            result.error  = str(e)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    def run_batch(self, tic_ids: list, sector: int = None) -> pd.DataFrame:
        log.info(f"Batch run: {len(tic_ids)} targets")
        all_results = []

        for tic in tic_ids:
            r = self.analyse_star(tic, sector=sector)
            self.results.append(r)
            all_results.append(r)

            if r.status in ('done', 'done_variability'):
                try:
                    plot_path  = self.viz.plot_result(r, self.output_dir)
                    r.plot_path = str(plot_path)
                except Exception as e:
                    log.warning(f"  Plot failed TIC {tic}: {e}")

        # Save JSON + CSV
        summary = [r.to_dict() for r in all_results]
        with open(self.output_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        df = pd.DataFrame(summary)
        df.to_csv(self.output_dir / 'results_table.csv', index=False)

        # Generate 3-page report
        try:
            self.rep.generate(all_results, self.output_dir)
        except Exception as e:
            log.warning(f"Report generation failed: {e}")

        return df

    def get_candidates(self):
        return [r for r in self.results
                if r.classification == 'planet_transit'
                and (r.confidence or 0) >= self.config.get('confidence_threshold', 0.70)]


# ─────────────────────────────────────────────────────────────────────────────
def _default_config() -> dict:
    return {
        'min_snr': 7.0, 'min_transit_count': 3,
        'period_min': 0.5, 'period_max': 13.0,
        'sigma_clip': 5.0, 'wotan_window': 0.75, 'wotan_method': 'biweight',
        'mcmc_walkers': 32, 'mcmc_steps': 3000, 'mcmc_burnin': 500,
        'confidence_threshold': 0.70,
        'output_dir': 'outputs', 'model_path': 'models/classifier.h5',
        'run_mcmc': False, 'run_centroid': True,
        'run_variability_screen': True,
        'rotation_period_min': 0.1, 'rotation_period_max': 30.0,
        'rotation_fap_threshold': 1e-3,
        'spot_amp_ratio_threshold': 5.0, 'spot_duty_cycle_threshold': 0.25,
        'run_blend_crosscheck': True,
        'contamination_ratio_threshold': 0.10,
        'blend_search_radius_px': 2.0, 'blend_max_neighbor_mag_diff': 4.0,
        'run_fap_bootstrap': True, 'fap_n_bootstrap': 100, 'random_seed': 42,
        'run_multiplanet_search': True,
        'max_planets_per_target': 3, 'multiplanet_sde_threshold': 7.0,
        'cache_dir': 'data/cache',
    }
