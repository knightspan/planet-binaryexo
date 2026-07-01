"""
run_pipeline.py
================
Single entry point for the JyotirVega Exoplanet Pipeline.
ISRO BAH 2026 — Problem Statement 7 | Team JyotirVega | Aurixys

Usage:
  python run_pipeline.py --demo                          # synthetic, no internet
  python run_pipeline.py --demo --mcmc                   # with MCMC uncertainties
  python run_pipeline.py --tics 261136679 388857263      # real TESS targets
  python run_pipeline.py --sector 1 --max-targets 50    # full sector batch
  python run_pipeline.py --real-demo                     # known TESS targets
"""

import argparse
import sys
import logging
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s [%(levelname)s]  %(message)s',
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/pipeline.log', mode='w',
                            encoding='utf-8'),
    ]
)
log = logging.getLogger(__name__)

# ─── Default config ────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    'min_snr':                    7.0,
    'min_transit_count':          3,
    'period_min':                 0.5,
    'period_max':                 13.0,
    'sigma_clip':                 5.0,
    'wotan_window':               0.75,
    'wotan_method':               'biweight',
    'mcmc_walkers':               32,
    'mcmc_steps':                 3000,
    'mcmc_burnin':                500,
    'confidence_threshold':       0.70,
    'output_dir':                 'outputs',
    'model_path':                 'models/classifier.h5',
    'run_mcmc':                   False,
    'run_centroid':               True,
    'run_variability_screen':     True,
    'rotation_period_min':        0.1,
    'rotation_period_max':        30.0,
    'rotation_fap_threshold':     1e-3,
    'spot_amp_ratio_threshold':   5.0,
    'spot_duty_cycle_threshold':  0.25,
    'run_blend_crosscheck':       True,
    'contamination_ratio_threshold': 0.10,
    'blend_search_radius_px':     2.0,
    'blend_max_neighbor_mag_diff':4.0,
    'run_fap_bootstrap':          True,
    'fap_n_bootstrap':            100,
    'random_seed':                42,
    'run_multiplanet_search':     True,
    'max_planets_per_target':     3,
    'multiplanet_sde_threshold':  7.0,
    'cache_dir':                  'data/cache',
}

# Synthetic demo targets — 4 classes, multiple seeds each
DEMO_SYNTHETIC = [
    ('planet_transit',      12345),
    ('planet_transit',      77777),
    ('eclipsing_binary',    54321),
    ('eclipsing_binary',    98765),
    ('blend',               99999),
    ('noise',               11111),
    ('noise',               22222),
]

# Known real TESS targets for the real-demo mode
REAL_DEMO_TICS = [
    261136679,   # HD 21749 — confirmed TESS planet
    388857263,   # known eclipsing binary
    100100827,   # TESS planet candidate
]


def main():
    Path('outputs').mkdir(exist_ok=True)
    Path('models').mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(
        description='JyotirVega Exoplanet Pipeline — ISRO BAH 2026 PS-7')
    parser.add_argument('--demo',         action='store_true',
                        help='Synthetic demo (no internet)')
    parser.add_argument('--real-demo',    action='store_true',
                        help='Real TESS known targets')
    parser.add_argument('--tics',         nargs='+', type=int,
                        help='Specific TIC IDs')
    parser.add_argument('--sector',       type=int, default=None)
    parser.add_argument('--max-targets',  type=int, default=20)
    parser.add_argument('--mcmc',         action='store_true')
    parser.add_argument('--no-fap',       action='store_true',
                        help='Skip bootstrap FAP (faster)')
    parser.add_argument('--no-variability',action='store_true',
                        help='Skip variability screen')
    parser.add_argument('--min-snr',      type=float, default=7.0)
    parser.add_argument('--period-max',   type=float, default=13.0)
    parser.add_argument('--fap-n',        type=int,   default=100)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config['run_mcmc']            = args.mcmc
    config['run_fap_bootstrap']   = not args.no_fap
    config['run_variability_screen'] = not args.no_variability
    config['min_snr']             = args.min_snr
    config['period_max']          = args.period_max
    config['fap_n_bootstrap']     = args.fap_n

    if args.demo:
        log.info("="*60)
        log.info("DEMO MODE — Synthetic light curves (no internet needed)")
        log.info("="*60)
        run_synthetic_demo(config)

    elif args.real_demo:
        log.info("REAL DEMO — Known TESS targets")
        run_batch(REAL_DEMO_TICS, config)

    elif args.tics:
        log.info(f"Analysing {len(args.tics)} user-specified TIC IDs")
        run_batch(args.tics, config, sector=args.sector)

    elif args.sector:
        run_sector(args.sector, args.max_targets, config)

    else:
        log.info("No mode specified — running synthetic demo. Use --help.")
        run_synthetic_demo(config)


# ── SYNTHETIC DEMO ─────────────────────────────────────────────────────────────
def run_synthetic_demo(config):
    import numpy as np
    import pandas as pd

    from pipeline.data_loader          import TESSDataLoader
    from pipeline.preprocessor         import LightCurvePreprocessor
    from pipeline.detector             import TransitDetector
    from pipeline.false_positive       import FalsePositiveVetter
    from pipeline.classifier           import ExoplanetClassifier, CLASSES
    from pipeline.fitter               import TransitParameterFitter
    from pipeline.snr_calculator       import SNRCalculator
    from pipeline.stellar_variability  import StellarVariabilityScreener
    from pipeline.blend_crosscheck     import BlendCrossChecker
    from pipeline.significance         import SignificanceEstimator, \
                                              SignificanceEstimator as SE
    from pipeline.multiplanet          import iterative_tls_search
    from pipeline.main_pipeline        import PipelineResult
    from utils.visualizer              import PipelineVisualizer
    from utils.report_generator        import ReportGenerator

    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    loader  = TESSDataLoader(config)
    pre     = LightCurvePreprocessor(config)
    det     = TransitDetector(config)
    vet     = FalsePositiveVetter(config)
    clf     = ExoplanetClassifier(config)
    fitter  = TransitParameterFitter(config)
    snr_c   = SNRCalculator()
    var_s   = StellarVariabilityScreener(config)
    sig     = SignificanceEstimator(config)
    viz     = PipelineVisualizer(config)
    rep     = ReportGenerator(config)

    all_results  = []
    summary_rows = []

    log.info(f"Running demo on {len(DEMO_SYNTHETIC)} synthetic targets\n")

    for true_cls, seed in DEMO_SYNTHETIC:
        log.info(f"{'─'*55}")
        log.info(f"  TIC {seed}  |  true class: {true_cls}")

        result = PipelineResult(seed)
        result.sector = 99

        try:
            # 1. Generate synthetic LC (physically realistic)
            lc = loader.get_synthetic_lc(seed, signal_type=true_cls)
            log.info(f"  LC generated: {len(lc['time'])} points  "
                     f"noise={np.std(lc['flux'])*1e6:.0f} ppm")

            # 2. Variability screen
            if config.get('run_variability_screen', True):
                var_r = var_s.screen(lc['time'], lc['flux'], lc['flux_err'])
                result.variability_screen = var_r
                if var_r['recommendation'] == 'route_to_variability_class':
                    log.info(f"  → stellar_variability "
                             f"(P={var_r.get('rotation_period',0):.2f}d)")
                    result.classification = 'stellar_variability'
                    result.confidence     = 0.90
                    result.class_probs    = {c: 0.0 for c in CLASSES}
                    result.snr = 0.0
                    result.status = 'done'
                    all_results.append(result)
                    correct = (true_cls in ('noise', 'stellar_variability'))
                    summary_rows.append(_row(seed, true_cls,
                                             'stellar_variability', correct,
                                             0.90, 0.0, 0.0, None))
                    continue

            # 3. Preprocess
            time, flux, flux_err = pre.process(lc)
            result.time     = time
            result.flux     = flux
            result.flux_err = flux_err

            # 4. TLS search
            tls = det.search(time, flux, flux_err)
            if tls is None or tls.SDE < 6.0:
                log.info(f"  No signal (SDE={getattr(tls,'SDE',0):.2f})")
                result.classification = 'noise'
                result.confidence     = 0.85
                result.class_probs    = {CLASSES[i]: [0.85,0.05,0.05,0.05][i]
                                         for i in range(4)}
                result.snr = 0.0; result.status = 'done'
                all_results.append(result)
                correct = (true_cls == 'noise')
                summary_rows.append(_row(seed, true_cls, 'noise',
                                          correct, 0.85, 0.0, 0.0, None))
                continue

            result.tls_result = tls
            log.info(f"  TLS: P={tls.period:.4f}d  "
                     f"depth={tls.depth*100:.4f}%  SDE={tls.SDE:.2f}")

            # Multi-planet search
            if config.get('run_multiplanet_search', True):
                planets = iterative_tls_search(
                    time, flux, flux_err, det,
                    max_planets=config.get('max_planets_per_target', 3),
                    sde_threshold=config.get('multiplanet_sde_threshold', 7.0))
                result.extra_planets = planets[1:]

            # 5. SNR
            snr = snr_c.compute(time, flux, flux_err, tls)
            result.snr = snr
            log.info(f"  SNR={snr:.2f}  [{snr_c.significance_statement(snr)}]")

            # 6. Bootstrap FAP
            fap_val = None
            if config.get('run_fap_bootstrap', True):
                fap_res = sig.bootstrap_fap(
                    time, flux, flux_err, det, tls.SDE,
                    n_bootstrap=config.get('fap_n_bootstrap', 100))
                fap_val = fap_res['fap']
                result.fap = fap_val
                log.info(f"  FAP={fap_val:.2e}  "
                         f"[{SE.significance_statement(fap_val)}]")

            # 7. Vetting
            flags = vet.vet(time, flux, flux_err, tls, run_centroid=False)
            result.vetting_flags   = flags
            result.odd_even_ratio  = flags.get('odd_even_ratio', 1.0)
            result.secondary_depth = flags.get('secondary_depth', 0.0)
            result.centroid_shift  = flags.get('centroid_shift', 0.0)

            # 8. Blend cross-check
            try:
                bc = BlendCrossChecker(config).check(
                    seed, tls.depth * 1e6,
                    centroid_shift_px=flags.get('centroid_shift', 0.0))
                flags['blend_crosscheck'] = bc
                result.blend_crosscheck   = bc
            except Exception:
                pass

            # 9. Classify
            label, conf, probs = clf.predict(time, flux, tls, flags)
            result.classification = label
            result.confidence     = conf
            result.class_probs    = {CLASSES[i]: float(probs[i]) for i in range(4)}
            correct = (label == true_cls)
            tick    = '✓' if correct else '✗'
            log.info(f"  {tick}  Predicted: {label} ({conf*100:.1f}%)  "
                     f"·  True: {true_cls}")

            # 10. Parameter fitting
            if label == 'planet_transit' and conf >= config['confidence_threshold']:
                tp, samples = fitter.fit(
                    time, flux, flux_err, tls,
                    run_mcmc=config.get('run_mcmc', False))
                result.transit_params = tp
                result.mcmc_samples   = samples
                if tp:
                    log.info(f"  Period:   {tp['period']:.5f} ± "
                             f"{tp['period_err']:.5f} d")
                    log.info(f"  Depth:    {tp['transit_depth']*100:.4f} ± "
                             f"{tp['transit_depth_err']*100:.4f} %")
                    log.info(f"  Duration: {tp['transit_duration']*24:.3f} ± "
                             f"{tp['transit_duration_err']*24:.3f} h")
                    log.info(f"  Rp/Rs:    {tp['rp_rs']:.5f} ± "
                             f"{tp['rp_rs_err']:.5f}"
                             f"  {'[MCMC]' if tp['mcmc_used'] else '[MAP]'}")

            result.status = 'done'
            all_results.append(result)
            summary_rows.append(_row(seed, true_cls, label, correct,
                                      conf, snr, tls.SDE, fap_val))

            # Plot
            try:
                plot_path = viz.plot_result(result, output_dir)
                result.plot_path = str(plot_path)
            except Exception as e:
                log.warning(f"  Plot failed: {e}")

        except Exception as e:
            log.error(f"  FAILED: {e}", exc_info=False)
            result.status = 'error'; result.error = str(e)
            all_results.append(result)

    # ── Summary ───────────────────────────────────────────────────────────────
    import pandas as pd
    df = pd.DataFrame(summary_rows)
    if len(df) == 0:
        log.warning("No results produced.")
        return df

    acc = df['correct'].mean() * 100
    n_c = int(df['correct'].sum())
    log.info(f"\n{'='*60}")
    log.info(f"DEMO COMPLETE  —  Accuracy: {acc:.1f}%  ({n_c}/{len(df)} correct)")
    log.info(f"\n{df[['tic_id','true_class','predicted','confidence','snr','sde','fap']].to_string(index=False)}")

    # Save outputs
    df.to_csv(output_dir / 'demo_results.csv', index=False)
    log.info(f"\nResults saved: {output_dir}/demo_results.csv")

    summary = [r.to_dict() for r in all_results]
    with open(output_dir / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Population plot
    try:
        viz.plot_population(all_results, output_dir)
    except Exception as e:
        log.warning(f"Population plot: {e}")

    # Report with REAL result objects (not stripped dummies)
    try:
        rep.generate(all_results, output_dir)
    except Exception as e:
        log.warning(f"Report: {e}")

    log.info(f"Plots saved:  {output_dir}/")
    return df


def _row(tic, true_cls, pred, correct, conf, snr, sde, fap):
    return {
        'tic_id':    tic,    'true_class': true_cls,
        'predicted': pred,   'correct':    bool(correct),
        'confidence': round(float(conf), 4),
        'snr':        round(float(snr),  2),
        'sde':        round(float(sde),  2),
        'fap':        fap,
    }


# ── BATCH (real TESS) ─────────────────────────────────────────────────────────
def run_batch(tic_ids: list, config: dict, sector: int = None):
    from pipeline.main_pipeline import ExoplanetPipeline
    from utils.visualizer       import PipelineVisualizer

    pipeline   = ExoplanetPipeline(config)
    output_dir = Path(config['output_dir'])
    viz        = PipelineVisualizer(config)

    df = pipeline.run_batch(tic_ids, sector=sector)

    for r in pipeline.get_candidates():
        if r.mcmc_samples is not None:
            viz.plot_corner(r.mcmc_samples, r.tic_id, output_dir)

    try:
        viz.plot_population(pipeline.results, output_dir)
    except Exception as e:
        log.warning(f"Population plot: {e}")

    log.info(f"\n{'='*60}")
    log.info("BATCH COMPLETE")
    if len(df) > 0:
        log.info(f"\n{df.to_string(index=False)}")
    log.info(f"Outputs: {output_dir}/")
    return df


# ── SECTOR BATCH ──────────────────────────────────────────────────────────────
def run_sector(sector: int, max_targets: int, config: dict):
    from pipeline.data_loader import TESSDataLoader
    loader  = TESSDataLoader(config)
    log.info(f"Downloading sector {sector} targets...")
    targets = loader.download_sector_batch(sector, max_targets=max_targets)
    if not targets:
        log.error("No targets found.")
        return
    tics = [t for t, _ in targets]
    log.info(f"Found {len(tics)} targets in sector {sector}")
    run_batch(tics, config, sector=sector)


if __name__ == '__main__':
    main()
