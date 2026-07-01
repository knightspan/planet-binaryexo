"""
dashboard/app.py
================
Interactive Streamlit web dashboard for the JyotirVega Exoplanet Pipeline.
ISRO BAH 2026 — Problem Statement 7

Features:
  - Single-star analysis (TIC ID input)
  - Batch analysis (multiple TIC IDs)
  - Synthetic demo (no internet needed)
  - Live pipeline run with real-time progress
  - Classification probabilities bar chart
  - Transit parameter table with uncertainties
  - FP vetting flag display
  - Light curve and phase-fold plots
  - Downloadable report

Run with:
  python -m streamlit run dashboard/app.py
  OR
  streamlit run dashboard/app.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
# Works for both: `streamlit run dashboard/app.py` (CWD = repo root)
# and Streamlit Cloud (CWD = repo root, __file__ = dashboard/app.py)
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also add CWD in case Streamlit Cloud sets CWD differently
CWD = Path(os.getcwd()).resolve()
if str(CWD) not in sys.path:
    sys.path.insert(0, str(CWD))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "JyotirVega · Exoplanet Pipeline",
    page_icon  = "🪐",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main { background: #07070D; }
  .stApp { background: #07070D; }
  h1, h2, h3 { color: #C9A84C !important; font-family: 'Courier New', monospace !important; }
  .metric-label { color: #888790 !important; font-size: 11px !important; }
  .stMetric { background: #12121A; border-radius: 8px; padding: 8px; }
  .badge { display:inline-block; padding:3px 12px; border-radius:12px;
           font-family:monospace; font-size:12px; font-weight:bold; }
  .bp { background:#1A2E4A; color:#4A90D9; }
  .be { background:#3A1A14; color:#E06C5A; }
  .bb { background:#2A1A3A; color:#9B72CF; }
  .bn { background:#1A1A24; color:#888790; }
  .bv { background:#0A2A1A; color:#2DC6B5; }
  div[data-testid="stExpander"] { background: #12121A; border: 1px solid #1E1E2C; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Config ───────────────────────────────────────────────────────────────────
BASE_CONFIG = {
    'min_snr': 7.0, 'min_transit_count': 3, 'period_min': 0.5,
    'period_max': 13.0, 'sigma_clip': 5.0, 'wotan_window': 0.75,
    'wotan_method': 'biweight', 'mcmc_walkers': 32, 'mcmc_steps': 3000,
    'mcmc_burnin': 500, 'confidence_threshold': 0.70,
    'output_dir': 'outputs', 'model_path': 'models/classifier.h5',
    'run_mcmc': False, 'run_centroid': False,
    'run_variability_screen': True,
    'rotation_period_min': 0.1, 'rotation_period_max': 30.0,
    'rotation_fap_threshold': 1e-3, 'spot_amp_ratio_threshold': 5.0,
    'spot_duty_cycle_threshold': 0.25,
    'run_blend_crosscheck': False,
    'contamination_ratio_threshold': 0.10, 'blend_search_radius_px': 2.0,
    'blend_max_neighbor_mag_diff': 4.0,
    'run_fap_bootstrap': True, 'fap_n_bootstrap': 15, 'random_seed': 42,
    'run_multiplanet_search': True, 'max_planets_per_target': 3,
    'multiplanet_sde_threshold': 7.0, 'cache_dir': 'data/cache',
    'tls_oversampling_factor': 1,   # fast for interactive dashboard (~10-15s);
                                    # full ISRO submissions use run_pipeline.py
                                    # directly with default factor=5
}

CNAMES = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']
CLASS_COLORS = {
    'noise': '#888790', 'planet_transit': '#4A90D9',
    'eclipsing_binary': '#E06C5A', 'blend': '#9B72CF',
    'stellar_variability': '#2DC6B5',
}
BADGE_CLASS = {
    'planet_transit': 'bp', 'eclipsing_binary': 'be',
    'blend': 'bb', 'noise': 'bn', 'stellar_variability': 'bv',
}
PAL = {
    'bg': '#0A0A0F', 'panel': '#12121A', 'grid': '#1E1E2C',
    'text': '#E8E8F0', 'accent': '#C9A84C', 'blue': '#4A90D9',
    'coral': '#E06C5A', 'grey': '#888790',
}

Path('outputs').mkdir(exist_ok=True)
Path('models').mkdir(exist_ok=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🪐 JyotirVega")
    st.markdown("**ISRO PS-7 · Exoplanet Pipeline**")
    st.markdown("*Aurixys · Team JyotirVega*")
    st.markdown("---")

    mode = st.radio("Mode", ["Synthetic Demo", "Single Star", "Batch Analysis"],
                    index=0)
    st.markdown("---")
    st.markdown("**Pipeline Settings**")

    min_snr    = st.slider("Min SNR",       4.0, 15.0, 7.0,  0.5)
    period_max = st.slider("Max period (d)", 3.0, 27.0, 13.0, 1.0)
    fap_n      = st.slider("FAP iterations (more = slower but more accurate)",
                           5, 100, 15, 5)
    run_mcmc   = st.checkbox("Run MCMC (slower, full posteriors)", False)
    run_var    = st.checkbox("Stellar variability screen", True)
    run_fap    = st.checkbox("Bootstrap FAP (adds ~30-60s)", False)

    st.markdown("---")
    st.caption("Pipeline: TLS · Wotan · DualCNN · emcee")
    st.caption("Team JyotirVega · Aurixys")

config = {**BASE_CONFIG,
          'min_snr': min_snr, 'period_max': period_max,
          'fap_n_bootstrap': fap_n, 'run_mcmc': run_mcmc,
          'run_variability_screen': run_var, 'run_fap_bootstrap': run_fap}




# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE RUNNER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _run_synthetic(seed: int, signal_type: str, config: dict):
    """Run pipeline on one synthetic light curve. Returns result dict."""
    try:
        from pipeline.data_loader         import TESSDataLoader
        from pipeline.preprocessor        import LightCurvePreprocessor
        from pipeline.detector            import TransitDetector
        from pipeline.false_positive      import FalsePositiveVetter
        from pipeline.classifier          import ExoplanetClassifier, CLASSES
        from pipeline.fitter              import TransitParameterFitter
        from pipeline.snr_calculator      import SNRCalculator
        from pipeline.stellar_variability import StellarVariabilityScreener
        from pipeline.significance        import SignificanceEstimator
        from pipeline.multiplanet         import iterative_tls_search

        loader = TESSDataLoader(config)
        pre    = LightCurvePreprocessor(config)
        det    = TransitDetector(config)
        vet    = FalsePositiveVetter(config)
        clf    = ExoplanetClassifier(config)
        fitter = TransitParameterFitter(config)
        snr_c  = SNRCalculator()
        var_s  = StellarVariabilityScreener(config)
        sig    = SignificanceEstimator(config)

        lc = loader.get_synthetic_lc(seed, signal_type=signal_type)

        # Variability screen
        var_r = var_s.screen(lc['time'], lc['flux'], lc['flux_err'])
        if var_r['recommendation'] == 'route_to_variability_class':
            return {
                'tic_id': seed, 'true_class': signal_type,
                'classification': 'stellar_variability',
                'confidence': 0.90, 'snr': 0.0, 'fap': None,
                'time': lc['time'], 'flux': lc['flux'],
                'tls': None, 'flags': {}, 'transit_params': None,
                'probs': {c: 0.0 for c in CNAMES},
                'variability': var_r,
            }

        time, flux, flux_err = pre.process(lc)
        tls = det.search(time, flux, flux_err)
        if tls is None or tls.SDE < 6.0:
            return {
                'tic_id': seed, 'true_class': signal_type,
                'classification': 'noise', 'confidence': 0.85,
                'snr': 0.0, 'fap': None,
                'time': time, 'flux': flux, 'tls': None,
                'flags': {}, 'transit_params': None,
                'probs': {'noise': 0.85, 'planet_transit': 0.05,
                          'eclipsing_binary': 0.05, 'blend': 0.05},
            }

        snr  = snr_c.compute(time, flux, flux_err, tls)
        fap  = None
        if config.get('run_fap_bootstrap', True):
            try:
                fap_res = sig.bootstrap_fap(
                    time, flux, flux_err, det, tls.SDE,
                    n_bootstrap=config.get('fap_n_bootstrap', 50))
                fap = fap_res['fap']
            except Exception:
                pass

        flags          = vet.vet(time, flux, flux_err, tls, run_centroid=False)
        label, conf, probs = clf.predict(time, flux, tls, flags)

        tp = None
        if label == 'planet_transit' and conf >= 0.70:
            tp, _ = fitter.fit(time, flux, flux_err, tls,
                               run_mcmc=config.get('run_mcmc', False))

        return {
            'tic_id': seed, 'true_class': signal_type,
            'classification': label, 'confidence': conf,
            'snr': snr, 'fap': fap,
            'time': time, 'flux': flux, 'flux_err': flux_err,
            'tls': tls, 'flags': flags, 'transit_params': tp,
            'probs': {CNAMES[i]: float(probs[i]) for i in range(4)},
        }
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return None


def _run_real(tic_id: int, sector, config: dict):
    """Run pipeline on a real TESS target."""
    try:
        from pipeline.main_pipeline import ExoplanetPipeline

        pipeline = ExoplanetPipeline(config)
        r        = pipeline.analyse_star(tic_id, sector=sector)

        if r.status in ('no_data', 'error', 'no_signal', 'low_snr'):
            st.warning(f"TIC {tic_id}: {r.status}"
                       + (f" — {r.error}" if r.error else ""))
            return None

        return {
            'tic_id':         r.tic_id,
            'true_class':     None,
            'classification': r.classification,
            'confidence':     r.confidence,
            'snr':            r.snr,
            'fap':            r.fap,
            'time':           r.time,
            'flux':           r.flux,
            'flux_err':       r.flux_err,
            'tls':            r.tls_result,
            'flags':          r.vetting_flags,
            'transit_params': r.transit_params,
            'probs':          r.class_probs or {c: 0.0 for c in CNAMES},
            'sector':         r.sector,
        }
    except Exception as e:
        st.error(f"Pipeline error TIC {tic_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _display_result(r: dict):
    """Render all result panels in Streamlit."""
    label = r.get('classification', 'unknown')
    conf  = r.get('confidence', 0.0) or 0.0
    snr   = r.get('snr', 0.0)
    fap   = r.get('fap')
    tls   = r.get('tls')
    tp    = r.get('transit_params')
    flags = r.get('flags', {})
    probs = r.get('probs', {})
    true_cls = r.get('true_class')
    col   = CLASS_COLORS.get(label, '#888790')
    badge = BADGE_CLASS.get(label, 'bn')

    # ── Header ───────────────────────────────────────────────────────────────
    correct_str = ""
    if true_cls:
        ok = label == true_cls
        correct_str = f"&nbsp;&nbsp;<em style='color:#888790'>True: {true_cls} {'✓' if ok else '✗'}</em>"

    st.markdown(f"""
    <div style="background:#12121A;border:1px solid {col};border-radius:10px;padding:18px;margin-bottom:12px">
      <span class="badge {badge}">{label.replace('_',' ').upper()}</span>
      &nbsp;&nbsp;
      <strong style="color:{col}">Confidence: {conf*100:.1f}%</strong>
      {correct_str}
    </div>
    """, unsafe_allow_html=True)

    # ── Metric row ────────────────────────────────────────────────────────────
    cols = st.columns(6)
    if tls:
        cols[0].metric("Period (d)",    f"{tls.period:.5f}")
        cols[1].metric("Depth (%)",     f"{tls.depth*100:.4f}")
        cols[2].metric("Duration (h)",  f"{tls.duration*24:.3f}")
        cols[3].metric("SDE (TLS)",     f"{tls.SDE:.2f}")
    cols[4].metric("SNR",           f"{snr:.1f}" if snr else "—")
    cols[5].metric("FAP",           f"{fap:.2e}" if fap else "—")

    # ── Plots ────────────────────────────────────────────────────────────────
    time = r.get('time')
    flux = r.get('flux')

    if time is not None and flux is not None and tls is not None:
        col_lc, col_pf = st.columns(2)

        with col_lc:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor(PAL['bg'])
            ax.set_facecolor(PAL['panel'])
            ax.plot(time, flux, '.', color=PAL['blue'], ms=0.7, alpha=0.5,
                    rasterized=True)
            if tls.transit_times:
                for tt in tls.transit_times:
                    ax.axvline(tt, color=col, alpha=0.4, lw=0.8, ls='--')
            ax.set_xlabel('Time (BTJD)', color=PAL['text'], fontsize=9)
            ax.set_ylabel('Normalised Flux', color=PAL['text'], fontsize=9)
            ax.set_title('Detrended Light Curve', color=PAL['accent'], fontsize=10)
            ax.tick_params(colors=PAL['text'])
            for sp in ax.spines.values(): sp.set_edgecolor(PAL['grid'])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with col_pf:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor(PAL['bg'])
            ax.set_facecolor(PAL['panel'])
            if tls.folded_phase is not None and tls.folded_y is not None:
                ax.plot(tls.folded_phase, tls.folded_y, '.', color=PAL['blue'],
                        ms=1.2, alpha=0.4, rasterized=True)
                if tls.model_folded_phase is not None:
                    ax.plot(tls.model_folded_phase, tls.model_folded_model,
                            color=col, lw=2, label='TLS model')
            else:
                ph = ((time - tls.T0) % tls.period) / tls.period
                ph[ph > 0.5] -= 1.0
                s  = np.argsort(ph)
                ax.plot(ph[s], flux[s], '.', color=PAL['blue'],
                        ms=0.8, alpha=0.4)
            ax.axhline(1.0, color=PAL['grey'], lw=0.5, ls='--')
            ax.set_xlim(-0.5, 0.5)
            ax.set_xlabel('Orbital Phase', color=PAL['text'], fontsize=9)
            ax.set_ylabel('Normalised Flux', color=PAL['text'], fontsize=9)
            ax.set_title(f'Phase-Folded  P={tls.period:.4f}d',
                         color=PAL['accent'], fontsize=10)
            ax.tick_params(colors=PAL['text'])
            for sp in ax.spines.values(): sp.set_edgecolor(PAL['grid'])
            ax.legend(fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # ── Classification probs + Vetting ────────────────────────────────────────
    col_prob, col_vet = st.columns(2)

    with col_prob:
        st.markdown("**Classification Probabilities**")
        if probs:
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor(PAL['bg'])
            ax.set_facecolor(PAL['panel'])
            vals   = [probs.get(c, 0) for c in CNAMES]
            colors = ['#888790', '#4A90D9', '#E06C5A', '#9B72CF']
            bars   = ax.bar(CNAMES, vals, color=colors, edgecolor=PAL['bg'],
                            width=0.6)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                        f'{v*100:.1f}%', ha='center', fontsize=8,
                        color=PAL['text'])
            ax.set_ylim(0, 1.15)
            ax.set_ylabel('Probability', color=PAL['text'], fontsize=9)
            ax.set_xticklabels([c.replace('_','\n') for c in CNAMES],
                               color=PAL['text'], fontsize=7)
            ax.tick_params(colors=PAL['text'])
            for sp in ax.spines.values(): sp.set_edgecolor(PAL['grid'])
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with col_vet:
        st.markdown("**FP Vetting Flags**")
        if flags:
            checks = [
                ("Odd/Even ratio",  flags.get('odd_even_ratio', 1.0),
                 lambda v: abs(v - 1.0) < 0.15),
                ("Secondary depth", flags.get('secondary_depth', 0.0),
                 lambda v: v < 0.001),
                ("Secondary SNR",   flags.get('secondary_snr', 0.0),
                 lambda v: v < 3.0),
                ("Centroid shift",  flags.get('centroid_shift', 0.0),
                 lambda v: v < 0.5),
                ("V-shape metric",  flags.get('v_shape_metric', 1.0),
                 lambda v: v > 0.6),
                ("N transits",      flags.get('n_transits', 0),
                 lambda v: v >= 3),
                ("FP score",        flags.get('fp_indicators', 0),
                 lambda v: v <= 1),
            ]
            for name, val, ok_fn in checks:
                ok  = ok_fn(val)
                ico = "✅" if ok else "⚠️"
                vs  = f"{val:.3f}" if isinstance(val, float) else str(val)
                st.markdown(f"{ico} **{name}**: `{vs}`")
        else:
            st.markdown("*No vetting data available*")

    # ── Transit parameters ────────────────────────────────────────────────────
    if tp:
        st.markdown("---")
        st.markdown("**Transit Parameters (batman model)**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Period (d)",     f"{tp['period']:.5f}",
                  f"±{tp['period_err']:.5f}")
        c2.metric("Depth (%)",      f"{tp['transit_depth']*100:.4f}",
                  f"±{tp['transit_depth_err']*100:.4f}")
        c3.metric("Duration (h)",   f"{tp['transit_duration']*24:.3f}",
                  f"±{tp['transit_duration_err']*24:.3f}")
        c4.metric("Rp/Rs",          f"{tp['rp_rs']:.5f}",
                  f"±{tp['rp_rs_err']:.5f}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("a/Rs",           f"{tp['a_rs']:.2f}",
                  f"±{tp['a_rs_err']:.2f}")
        c6.metric("Inclination (°)",f"{tp['inclination']:.2f}",
                  f"±{tp['inclination_err']:.2f}")
        c7.metric("Impact param b", f"{tp.get('impact_parameter', 0):.3f}")
        c8.metric("Fit method",     "MCMC" if tp['mcmc_used'] else "MAP")

        if tp['mcmc_used']:
            st.success("✓ Uncertainties from MCMC posterior (16th–84th percentile)")
        else:
            st.info("ℹ MAP uncertainties. Enable MCMC in sidebar for full posteriors.")


def _result_row(r: dict) -> dict:
    tls = r.get('tls')
    tp  = r.get('transit_params')
    return {
        'TIC':         r.get('tic_id', ''),
        'Class':       r.get('classification', ''),
        'Confidence':  f"{(r.get('confidence') or 0)*100:.1f}%",
        'SNR':         round(r.get('snr') or 0, 2),
        'FAP':         f"{r['fap']:.2e}" if r.get('fap') else '—',
        'Period_d':    round(tls.period, 5) if tls else '—',
        'Depth_pct':   round(tls.depth * 100, 4) if tls else '—',
        'Duration_h':  round(tls.duration * 24, 3) if tls else '—',
        'Rp/Rs':       round(tp['rp_rs'], 5) if tp else '—',
    }


def _parse_tics(text: str):
    tics = []
    for part in text.replace(',', '\n').split('\n'):
        part = part.strip()
        if part.isdigit():
            tics.append(int(part))
    return tics


# ─── Header ───────────────────────────────────────────────────────────────────
st.title("🪐 AI Exoplanet Detection Pipeline")
st.markdown("*ISRO BAH 2026 · Problem Statement 7 · JyotirVega · Aurixys*")
st.markdown("---")

# ─── MODE: SYNTHETIC DEMO ─────────────────────────────────────────────────────
if mode == "Synthetic Demo":
    st.markdown("## Synthetic Demo")
    st.info("Runs the complete 10-stage pipeline on physically realistic "
            "synthetic TESS light curves. No internet connection needed.")

    col1, col2, col3 = st.columns(3)
    with col1:
        signal_type = st.selectbox(
            "Signal type",
            ['planet_transit', 'eclipsing_binary', 'blend', 'noise',
             'stellar_variability'])
    with col2:
        seed = st.number_input("Seed (TIC-like ID)", 1, 999999, 12345)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Run Pipeline", type="primary",
                            use_container_width=True)

    if run_btn:
        with st.spinner("Running full 10-stage pipeline..."):
            result = _run_synthetic(int(seed), signal_type, config)
        if result:
            _display_result(result)

# ─── MODE: SINGLE STAR ────────────────────────────────────────────────────────
elif mode == "Single Star":
    st.markdown("## Single Star Analysis (Real TESS Data)")
    st.info("Downloads a real TESS light curve from MAST and runs the full pipeline.")

    col1, col2, col3 = st.columns(3)
    with col1:
        tic_id = st.number_input("TIC ID", 1, 999999999, 261136679)
    with col2:
        sector = st.number_input("Sector (0 = any)", 0, 70, 0)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶ Analyse Star", type="primary",
                            use_container_width=True)

    st.markdown("**Known test targets:**")
    st.markdown("""
    | TIC ID | Object | Expected class |
    |--------|--------|----------------|
    | `261136679` | HD 21749 | planet_transit |
    | `388857263` | Known EB | eclipsing_binary |
    | `100100827` | TOI candidate | planet_transit |
    """)

    if run_btn:
        sector_arg = int(sector) if sector > 0 else None
        with st.spinner(f"Downloading and analysing TIC {tic_id}..."):
            result = _run_real(int(tic_id), sector_arg, config)
        if result:
            _display_result(result)

# ─── MODE: BATCH ──────────────────────────────────────────────────────────────
elif mode == "Batch Analysis":
    st.markdown("## Batch Analysis")
    st.info("Analyse multiple TIC IDs. For real TESS data, provide IDs one per line.")

    tic_text = st.text_area(
        "TIC IDs (one per line or comma-separated)",
        value="261136679\n388857263\n100100827", height=120)

    run_btn = st.button("▶ Run Batch", type="primary")
    if run_btn:
        tic_ids = _parse_tics(tic_text)
        st.info(f"Running on {len(tic_ids)} targets...")
        pb      = st.progress(0)
        rows    = []
        results = []

        for i, tic in enumerate(tic_ids):
            pb.progress((i + 1) / len(tic_ids))
            with st.spinner(f"[{i+1}/{len(tic_ids)}] TIC {tic}..."):
                r = _run_real(tic, None, config)
            if r:
                results.append(r)
                rows.append(_result_row(r))

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("⬇ Download CSV", csv,
                               "jyotirvega_results.csv", "text/csv")

        if results:
            cands = [r for r in results
                     if r.get('classification') == 'planet_transit'
                     and (r.get('confidence') or 0) >= 0.70]
            if cands:
                st.success(f"✓ {len(cands)} planet transit candidate(s) found!")
                for r in cands:
                    st.markdown(f"**TIC {r['tic_id']}**: "
                                f"P={r.get('period_d','?')}d  "
                                f"depth={r.get('depth_pct','?')}%  "
                                f"conf={r.get('confidence_pct','?')}")


