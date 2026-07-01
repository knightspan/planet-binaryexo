"""
utils/visualizer.py
====================
Generates all visualization outputs:
  1. 6-panel per-star diagnostic plot
  2. MCMC corner plot (posterior distributions)
  3. Population summary (period-depth scatter + SNR histogram)
"""

import numpy as np
import logging
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

log = logging.getLogger(__name__)

PAL = {
    'bg':      '#0A0A0F', 'panel':  '#12121A', 'grid':   '#1E1E2C',
    'text':    '#E8E8F0', 'accent': '#C9A84C', 'blue':   '#4A90D9',
    'coral':   '#E06C5A', 'purple': '#9B72CF', 'teal':   '#2DC6B5',
    'green':   '#5AB88A', 'grey':   '#888790',
    'classes': ['#888790', '#4A90D9', '#E06C5A', '#9B72CF'],
}
CNAMES = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']


def _style():
    plt.rcParams.update({
        'figure.facecolor': PAL['bg'],  'axes.facecolor':  PAL['panel'],
        'axes.edgecolor':   PAL['grid'],'axes.labelcolor': PAL['text'],
        'text.color':       PAL['text'],'xtick.color':     PAL['text'],
        'ytick.color':      PAL['text'],'grid.color':      PAL['grid'],
        'grid.alpha':       0.35,       'font.family':     'monospace',
    })


class PipelineVisualizer:

    def __init__(self, config: dict):
        self.config = config

    # ─────────────────────────────────────────────────────────────────────────
    # 6-PANEL DIAGNOSTIC PLOT
    # ─────────────────────────────────────────────────────────────────────────
    def plot_result(self, result, output_dir: Path) -> Path:
        _style()
        fig = plt.figure(figsize=(20, 13))
        fig.patch.set_facecolor(PAL['bg'])
        gs  = gridspec.GridSpec(3, 3, figure=fig,
                                hspace=0.45, wspace=0.32,
                                left=0.06, right=0.97, top=0.91, bottom=0.07)

        tic   = result.tic_id
        label = result.classification or 'unknown'
        conf  = result.confidence or 0.0
        tls   = result.tls_result
        tp    = result.transit_params

        col = PAL['classes'][CNAMES.index(label)] \
              if label in CNAMES else PAL['grey']

        fig.suptitle(
            f'TIC {tic}   ·   {label.replace("_"," ").upper()}'
            f'   ·   Confidence {conf*100:.1f}%'
            f'{"  ·  SNR=" + str(round(result.snr, 1)) if result.snr else ""}',
            fontsize=13, fontweight='bold', color=col, y=0.96)

        time = result.time
        flux = result.flux

        if time is None or tls is None:
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, 'Insufficient data for plotting',
                    ha='center', va='center', color=PAL['grey'], fontsize=14)
            path = output_dir / f'TIC_{tic}_diagnostic.png'
            fig.savefig(path, dpi=100, bbox_inches='tight',
                        facecolor=PAL['bg'])
            plt.close(fig)
            return path

        # ── Panel 1: Full detrended LC ───────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(time, flux, '.', color=PAL['blue'], ms=0.7, alpha=0.5,
                 rasterized=True)
        if tls and tls.transit_times:
            for tt in tls.transit_times:
                ax1.axvline(tt, color=col, alpha=0.4, lw=0.8, ls='--')
        ax1.set_xlabel('Time (BTJD)', fontsize=9)
        ax1.set_ylabel('Normalised Flux', fontsize=9)
        ax1.set_title('Detrended Light Curve (Wotan biweight)', fontsize=10,
                      color=PAL['accent'])
        ax1.grid(True, alpha=0.25)
        ax1.text(0.01, 0.97, f"Sector {result.sector or '?'}  "
                              f"·  {len(time):,} pts",
                 transform=ax1.transAxes, fontsize=7,
                 color=PAL['grey'], va='top')

        # ── Panel 2: TLS periodogram ─────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        if tls.periods is not None and tls.power is not None:
            ax2.plot(tls.periods, tls.power, color=PAL['blue'], lw=0.7, alpha=0.8)
            ax2.axvline(tls.period, color=col, lw=1.8,
                        label=f'P={tls.period:.4f}d')
            for h in [2, 3]:
                ax2.axvline(tls.period / h, color=PAL['grey'], lw=0.6,
                            ls=':', alpha=0.5)
            ax2.set_xlabel('Period (days)', fontsize=9)
            ax2.set_ylabel('TLS Power (SDE)', fontsize=9)
            ax2.set_title(f'TLS Periodogram  SDE={tls.SDE:.2f}', fontsize=10,
                          color=PAL['accent'])
            ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.25)

        # ── Panel 3: Phase-folded (all data) ─────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :2])
        if tls.folded_phase is not None and tls.folded_y is not None:
            ax3.plot(tls.folded_phase, tls.folded_y, '.', color=PAL['blue'],
                     ms=1.0, alpha=0.35, rasterized=True)
            if tls.model_folded_phase is not None:
                ax3.plot(tls.model_folded_phase, tls.model_folded_model,
                         color=col, lw=2.0, label='TLS model', zorder=5)
        else:
            phase = ((time - tls.T0) % tls.period) / tls.period
            phase[phase > 0.5] -= 1.0
            s = np.argsort(phase)
            ax3.plot(phase[s], flux[s], '.', color=PAL['blue'],
                     ms=0.7, alpha=0.35, rasterized=True)
        ax3.axhline(1.0, color=PAL['grey'], lw=0.5, ls='--')
        ax3.set_xlim(-0.5, 0.5)
        ax3.set_xlabel('Orbital Phase', fontsize=9)
        ax3.set_ylabel('Normalised Flux', fontsize=9)
        ax3.set_title(f'Phase-Folded  P={tls.period:.5f}d', fontsize=10,
                      color=PAL['accent'])
        ax3.legend(fontsize=7)
        ax3.grid(True, alpha=0.25)

        # ── Panel 4: Transit zoom + batman model ─────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        phase = ((time - tls.T0) % tls.period) / tls.period
        phase[phase > 0.5] -= 1.0
        zoom  = np.abs(phase) < 0.075
        if zoom.sum() > 5:
            s = np.argsort(phase[zoom])
            ax4.plot(phase[zoom][s], flux[zoom][s], '.',
                     color=PAL['coral'], ms=2.5, alpha=0.7)
        # batman overlay if params available
        if tp is not None:
            bm_phase, bm_flux = _batman_curve(
                tp['period'], tp['t0'], tp['rp_rs'], tp['a_rs'], tp['inclination'])
            if bm_phase is not None:
                ax4.plot(bm_phase, bm_flux, color=PAL['accent'], lw=2.2,
                         label='batman fit', zorder=5)
        ax4.axhline(1.0, color=PAL['grey'], lw=0.5, ls='--')
        ax4.set_xlabel('Orbital Phase', fontsize=9)
        ax4.set_ylabel('Normalised Flux', fontsize=9)
        ax4.set_title('Transit Zoom + batman Model', fontsize=10,
                      color=PAL['accent'])
        ax4.legend(fontsize=7)
        ax4.grid(True, alpha=0.25)

        # ── Panel 5: Classification probabilities ─────────────────────────────
        ax5 = fig.add_subplot(gs[2, 0])
        if result.class_probs:
            probs  = [result.class_probs.get(c, 0) for c in CNAMES]
            bars   = ax5.bar(CNAMES, probs, color=PAL['classes'],
                             edgecolor=PAL['bg'], width=0.6)
            for bar, p in zip(bars, probs):
                ax5.text(bar.get_x() + bar.get_width()/2, p + 0.01,
                         f'{p*100:.1f}%', ha='center', fontsize=8,
                         color=PAL['text'])
            ax5.set_ylim(0, 1.15)
            ax5.set_ylabel('Probability', fontsize=9)
            ax5.set_title('Classification', fontsize=10, color=PAL['accent'])
            ax5.set_xticklabels([c.replace('_', '\n') for c in CNAMES], fontsize=7)
            ax5.grid(True, alpha=0.2, axis='y')

        # ── Panel 6: Parameter table ──────────────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')
        _draw_params(ax6, tls, tp, result)

        # ── Panel 7: Vetting flags ─────────────────────────────────────────────
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        _draw_vetting(ax7, result.vetting_flags)

        path = output_dir / f'TIC_{tic}_diagnostic.png'
        fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=PAL['bg'])
        plt.close(fig)
        log.info(f"  Plot saved: {path.name}")
        return path

    # ─────────────────────────────────────────────────────────────────────────
    # MCMC CORNER PLOT
    # ─────────────────────────────────────────────────────────────────────────
    def plot_corner(self, samples, tic_id, output_dir: Path):
        try:
            import corner
            _style()
            labels = [r'$t_0$ (BTJD)', r'$R_p/R_\star$', r'$a/R_\star$',
                      r'Inc (°)']
            fig = corner.corner(
                samples, labels=labels,
                quantiles=[0.16, 0.50, 0.84], show_titles=True,
                title_kwargs={'fontsize': 10}, color=PAL['accent'],
                label_kwargs={'fontsize': 9})
            fig.patch.set_facecolor(PAL['bg'])
            for ax in fig.get_axes():
                ax.set_facecolor(PAL['panel'])
            fig.suptitle(f'TIC {tic_id} — MCMC Posterior',
                         color=PAL['accent'], fontsize=12)
            path = output_dir / f'TIC_{tic_id}_corner.png'
            fig.savefig(path, dpi=100, bbox_inches='tight',
                        facecolor=PAL['bg'])
            plt.close(fig)
            return path
        except Exception as e:
            log.debug(f"Corner plot failed: {e}")
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # POPULATION SUMMARY
    # ─────────────────────────────────────────────────────────────────────────
    def plot_population(self, results, output_dir: Path):
        _style()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor(PAL['bg'])
        fig.suptitle('Population Summary — All Analysed Targets',
                     color=PAL['accent'], fontsize=13)

        for ax in axes:
            ax.set_facecolor(PAL['panel'])

        cls_data = {c: {'periods': [], 'depths': [], 'snrs': []}
                    for c in CNAMES}

        for r in results:
            if r.tls_result is None or r.classification is None:
                continue
            c = r.classification if r.classification in CNAMES else 'noise'
            cls_data[c]['periods'].append(r.tls_result.period)
            cls_data[c]['depths'].append(r.tls_result.depth * 100)
            cls_data[c]['snrs'].append(r.snr or 0)

        # Period vs Depth
        for c, col in zip(CNAMES, PAL['classes']):
            if cls_data[c]['periods']:
                axes[0].scatter(cls_data[c]['periods'], cls_data[c]['depths'],
                                c=col, label=c.replace('_', ' '),
                                alpha=0.8, s=50, edgecolors='none')
        axes[0].set_xlabel('Period (days)', fontsize=10)
        axes[0].set_ylabel('Transit Depth (%)', fontsize=10)
        axes[0].set_title('Period vs Transit Depth', fontsize=11,
                          color=PAL['accent'])
        axes[0].set_yscale('log')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.25)

        # SNR histogram
        for c, col in zip(CNAMES, PAL['classes']):
            if cls_data[c]['snrs']:
                axes[1].hist(cls_data[c]['snrs'], bins=15, color=col,
                             alpha=0.6, label=c.replace('_', ' '),
                             edgecolor='none')
        axes[1].axvline(7, color=PAL['accent'], ls='--', lw=1.5,
                        label='SNR=7 threshold')
        axes[1].set_xlabel('SNR', fontsize=10)
        axes[1].set_ylabel('Count', fontsize=10)
        axes[1].set_title('SNR Distribution', fontsize=11, color=PAL['accent'])
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.25)

        path = output_dir / 'population_summary.png'
        fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=PAL['bg'])
        plt.close(fig)
        log.info(f"Population plot saved: {path.name}")
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _batman_curve(period, t0, rp, a, inc):
    try:
        import batman
        p = batman.TransitParams()
        p.t0 = 0.0; p.per = period; p.rp = rp; p.a = a; p.inc = inc
        p.ecc = 0.0; p.w = 90.0; p.u = [0.40, 0.26]; p.limb_dark = "quadratic"
        phase = np.linspace(-0.075, 0.075, 500)
        m     = batman.TransitModel(p, phase * period)
        return phase, m.light_curve(p)
    except Exception:
        return None, None


def _draw_params(ax, tls, tp, result):
    ax.set_facecolor(PAL['panel'])
    ax.set_title('Transit Parameters', fontsize=10, color=PAL['accent'])
    rows = [('Parameter', 'Value', '±')]
    if tls:
        rows += [
            ('Period (d)',      f'{tls.period:.5f}',        f'{tls.period_uncertainty:.5f}'),
            ('Depth (%)',       f'{tls.depth*100:.4f}',     '—'),
            ('Duration (h)',    f'{tls.duration*24:.3f}',   '—'),
            ('SDE',            f'{tls.SDE:.2f}',            '—'),
            ('SNR',            f'{result.snr:.2f}' if result.snr else '—', '—'),
            ('FAP',            f'{result.fap:.2e}' if result.fap else '—', '—'),
            ('N transits',     str(tls.transit_count or 0), '—'),
        ]
    if tp:
        rows += [
            ('Rp/Rs',          f'{tp["rp_rs"]:.5f}',       f'{tp["rp_rs_err"]:.5f}'),
            ('a/Rs',           f'{tp["a_rs"]:.2f}',         f'{tp["a_rs_err"]:.2f}'),
            ('Inc (°)',        f'{tp["inclination"]:.2f}',  f'{tp["inclination_err"]:.2f}'),
            ('b (impact)',     f'{tp.get("impact_parameter",0):.3f}', '—'),
            ('Fit method',     'MCMC' if tp['mcmc_used'] else 'MAP', '—'),
        ]

    y0, dy = 0.98, 0.082
    for j, (lbl, val, err) in enumerate(rows):
        y   = y0 - j * dy
        bold = j == 0
        c    = PAL['accent'] if bold else PAL['text']
        fw   = 'bold' if bold else 'normal'
        ax.text(0.02, y, lbl, transform=ax.transAxes, fontsize=8,
                color=c, fontweight=fw, va='top')
        ax.text(0.50, y, val, transform=ax.transAxes, fontsize=8,
                color=c, fontweight=fw, va='top')
        ax.text(0.78, y, err, transform=ax.transAxes, fontsize=7,
                color=PAL['grey'], va='top')


def _draw_vetting(ax, flags):
    ax.set_facecolor(PAL['panel'])
    ax.set_title('FP Vetting Flags', fontsize=10, color=PAL['accent'])
    if not flags:
        ax.text(0.5, 0.5, 'No vetting data', ha='center', va='center',
                color=PAL['grey'], fontsize=9, transform=ax.transAxes)
        return

    checks = [
        ('Odd/Even ratio',   flags.get('odd_even_ratio', 1.0),
         lambda v: abs(v - 1.0) < 0.15, '|ratio−1| < 0.15 = OK'),
        ('Secondary depth',  flags.get('secondary_depth', 0.0),
         lambda v: v < 0.001, '< 0.001 = OK'),
        ('Secondary SNR',    flags.get('secondary_snr', 0.0),
         lambda v: v < 3.0,  '< 3 = OK'),
        ('Centroid shift',   flags.get('centroid_shift', 0.0),
         lambda v: v < 0.5,  '< 0.5 px = OK'),
        ('V-shape metric',   flags.get('v_shape_metric', 1.0),
         lambda v: v > 0.6,  '> 0.6 = flat-bottom'),
        ('N transits',       flags.get('n_transits', 0),
         lambda v: v >= 3,   '≥ 3 = OK'),
        ('FP score',         flags.get('fp_indicators', 0),
         lambda v: v <= 1,   '≤ 1/5 = OK'),
    ]

    y0, dy = 0.93, 0.12
    for j, (name, val, ok_fn, hint) in enumerate(checks):
        y  = y0 - j * dy
        ok = ok_fn(val)
        c  = PAL['green'] if ok else PAL['coral']
        vs = f'{val:.3f}' if isinstance(val, float) else str(val)
        ax.text(0.02, y,       f'{"✓" if ok else "✗"}  {name}',
                transform=ax.transAxes, fontsize=8, color=c, va='top')
        ax.text(0.72, y,       vs,
                transform=ax.transAxes, fontsize=8, color=PAL['text'], va='top')
        ax.text(0.72, y-dy*0.45, hint,
                transform=ax.transAxes, fontsize=6.5, color=PAL['grey'], va='top')
