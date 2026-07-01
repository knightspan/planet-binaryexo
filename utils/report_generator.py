"""
utils/report_generator.py
==========================
Auto-generates the 3-page HTML report required by ISRO PS-7.

Covers:
  Page 1 — Methodology (TLS, Wotan, dual-view CNN, MCMC, calibration)
  Page 2 — Tools & libraries table + data sources
  Page 3 — Results: candidate table, parameter estimates, uncertainties
"""

import logging
import json
from pathlib import Path
from datetime import datetime

log = logging.getLogger(__name__)

CNAMES = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']

CSS = """
<style>
body{background:#07070D;color:#E8E8F0;font-family:'Segoe UI',Georgia,serif;
     font-size:13.5px;line-height:1.75;max-width:920px;margin:0 auto;padding:36px 28px}
h1{font-family:'Courier New',monospace;color:#C9A84C;font-size:20px;
   border-bottom:1px solid #C9A84C;padding-bottom:8px;margin-bottom:18px}
h2{font-family:'Courier New',monospace;color:#C9A84C;font-size:15px;margin:28px 0 10px}
h3{color:#4A90D9;font-size:13px;margin:18px 0 7px}
p{margin-bottom:9px}
code{font-family:'Courier New',monospace;font-size:11px;background:#12121A;
     padding:1px 6px;border-radius:3px;color:#C9A84C}
table{width:100%;border-collapse:collapse;margin:14px 0;font-size:12px}
th{background:#12121A;color:#C9A84C;padding:7px 9px;text-align:left}
td{padding:6px 9px;border-bottom:1px solid #1E1E2C}
.p{color:#4A90D9;font-weight:bold} .e{color:#E06C5A;font-weight:bold}
.b{color:#9B72CF;font-weight:bold} .n{color:#888790}
.box{background:#12121A;border-left:3px solid #C9A84C;
     padding:11px 15px;margin:14px 0;border-radius:2px}
hr{border:none;border-top:1px solid #1E1E2C;margin:28px 0}
.meta{color:#888790;font-size:11px;font-family:'Courier New',monospace}
</style>
"""


class ReportGenerator:

    def __init__(self, config: dict):
        self.config = config

    def generate(self, results: list, output_dir: Path):
        output_dir = Path(output_dir)

        candidates = [r for r in results
                      if r.classification == 'planet_transit'
                      and (r.confidence or 0) >= self.config.get(
                          'confidence_threshold', 0.70)]
        ebs        = [r for r in results
                      if r.classification == 'eclipsing_binary']
        blends     = [r for r in results
                      if r.classification == 'blend']
        var_stars  = [r for r in results
                      if r.classification == 'stellar_variability']
        noise_n    = sum(1 for r in results if r.classification == 'noise')
        done_n     = sum(1 for r in results
                         if r.status in ('done', 'done_variability'))

        html = self._render(results, candidates, ebs, blends,
                            var_stars, noise_n, done_n)

        html_path = output_dir / 'report.html'
        html_path.write_text(html, encoding='utf-8')
        log.info(f"Report saved: {html_path}")

        md_path = output_dir / 'report.md'
        md_path.write_text(self._markdown(results, candidates), encoding='utf-8')
        log.info(f"Markdown report saved: {md_path}")
        return html_path

    # ─────────────────────────────────────────────────────────────────────────
    def _render(self, results, candidates, ebs, blends, var_stars,
                noise_n, done_n):
        ts  = datetime.now().strftime('%Y-%m-%d %H:%M UTC')
        mcmc_used = any(r.transit_params and r.transit_params.get('mcmc_used')
                        for r in candidates)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>ISRO PS-7 — JyotirVega Exoplanet Report</title>
{CSS}</head>
<body>
<h1>AI-Enabled Exoplanet Detection from Noisy TESS Light Curves</h1>
<p class="meta">ISRO Hackathon 2026 &nbsp;·&nbsp; Problem Statement 7 &nbsp;·&nbsp;
Team JyotirVega &nbsp;·&nbsp; Aurixys &nbsp;·&nbsp; Generated: {ts}</p>

<hr>
<h2>1 &nbsp; Methodology</h2>

<h3>1.1 &nbsp; Pipeline Overview</h3>
<div class="box">
<strong>10-stage pipeline:</strong> &nbsp;
TESS MAST Download &rarr;
Variability Pre-screen (Lomb-Scargle) &rarr;
Wotan Biweight Detrend &rarr;
TLS Period Search &rarr;
SNR Computation &rarr;
Bootstrap FAP &rarr;
5-test FP Vetting &rarr;
Blend Cross-check (TIC &plus; Gaia) &rarr;
Dual-view CNN Classification &rarr;
batman &plus; emcee MCMC Fitting
</div>

<h3>1.2 &nbsp; Detrending (Wotan Biweight)</h3>
<p>Stellar variability and instrumental trends are removed using Wotan's
biweight location estimator (Hippke et al. 2019) with a 0.75-day sliding
window. Unlike Savitzky-Golay, the biweight estimator is robust to outliers
and avoids polynomial ringing artifacts that introduce spurious dips.
Break tolerance of 0.5 days correctly handles momentum-dump gaps.</p>

<h3>1.3 &nbsp; Transit Detection (TLS vs BLS)</h3>
<p>Transit Least Squares (Hippke &amp; Heller 2019) uses a physically accurate
limb-darkened transit template with quadratic coefficients
<code>u&sub;1;=0.40, u&sub;2;=0.26</code> (TESS G-band, G-type stars).
This gives ~10&ndash;15% better sensitivity to small planets than BLS, which
models transits as rectangular boxes. Period uncertainty is estimated from
the SDE peak width. An iterative mask-and-re-search step recovers additional
planets in multi-planet systems.</p>

<h3>1.4 &nbsp; Stellar Variability Screen (Starspots)</h3>
<p>Before any transit search, a Lomb-Scargle periodogram is run on the
raw light curve. Targets with a statistically significant periodicity
(FAP &lt; 0.1%), high amplitude ratio (&gt;5&times; noise floor), and broad
phase coverage (duty cycle &gt;25%) are routed to the
<code>stellar_variability</code> class and never enter the 4-class CNN.
This directly addresses the PS-7 objective text's explicit mention of
starspots as a confound.</p>

<h3>1.5 &nbsp; False-Positive Vetting (5 Independent Tests)</h3>
<table>
<tr><th>Test</th><th>Physical signature</th><th>Pass criterion</th></tr>
<tr><td>Odd/even depth ratio</td><td>EB alternating eclipses</td><td>|ratio &minus; 1| &lt; 0.15</td></tr>
<tr><td>Secondary eclipse (phase 0.5)</td><td>EB secondary eclipse</td><td>SNR &lt; 3</td></tr>
<tr><td>Centroid shift (TPF)</td><td>Background binary</td><td>&lt; 0.5 px shift</td></tr>
<tr><td>V-shape metric</td><td>Grazing EB, no flat bottom</td><td>&gt; 0.6</td></tr>
<tr><td>Depth variability</td><td>Blend or EB</td><td>std/mean &lt; 0.3</td></tr>
</table>

<h3>1.6 &nbsp; Classification</h3>
<p>A dual-view 1D CNN (Shallue &amp; Vanderburg 2018 AstroNet architecture)
takes two phase-folded views as input:
<strong>Global view</strong> (2001 bins, phase &minus;0.5 to +0.5) captures
the full light curve context and secondary eclipses;
<strong>Local view</strong> (201 bins, phase &plusmn;0.075) zooms on the
transit to resolve ingress/egress curvature.
Two parallel CNN towers merge into a 4-class softmax head.
Output is combined with a physics-informed rule classifier in a 60/40
weighted ensemble.</p>

<h3>1.7 &nbsp; Parameter Estimation and Uncertainty Quantification</h3>
<p>Planet candidates are fit using a batman Mandel-Agol transit model
(Kreidberg 2015) in two stages:
(1) Nelder-Mead MAP optimisation for a fast initial best-fit;
(2) emcee ensemble MCMC sampler ({self.config.get('mcmc_walkers',32)} walkers
&times; {self.config.get('mcmc_steps',3000)} steps, {self.config.get('mcmc_burnin',500)}-step burn-in)
for full posterior distributions.
Uncertainties reported as 16th&ndash;84th percentile credible intervals.
Free parameters: <code>t&sub;0;, R&sub;p;/R&sub;&star;, a/R&sub;&star;, i</code>.
Fixed: eccentricity = 0, limb darkening from TESS bandpass tables.</p>

<h3>1.8 &nbsp; Significance (Bootstrap FAP)</h3>
<p>Each detection carries a permutation-bootstrap False Alarm Probability:
the flux array is shuffled {self.config.get('fap_n_bootstrap',100)} times,
TLS is re-run on each permutation, and FAP = fraction of permutations
achieving SDE &ge; the observed SDE. This is assumption-free and does not
rely on analytic SDE&ndash;FAP approximations.</p>

<h3>1.9 &nbsp; Assumptions</h3>
<ul style="margin:8px 0 12px 20px;line-height:1.9">
<li>Circular orbits assumed (e=0); appropriate for short-period TESS planets</li>
<li>Single-sector analysis (27 days); long-period planets not recoverable</li>
<li>Quadratic limb darkening with solar-like coefficients; may under-perform for M-dwarfs</li>
<li>CNN requires trained weights for best performance; rule-based ensemble used otherwise</li>
</ul>

<hr>
<h2>2 &nbsp; Tools and Libraries</h2>
<table>
<tr><th>Library</th><th>Version</th><th>Purpose</th></tr>
<tr><td><code>lightkurve</code></td><td>&ge;2.4</td><td>TESS LC download from MAST, TPF centroid</td></tr>
<tr><td><code>wotan</code></td><td>&ge;1.11</td><td>Biweight robust detrending</td></tr>
<tr><td><code>transitleastsquares</code></td><td>&ge;1.0.31</td><td>TLS limb-darkened transit search</td></tr>
<tr><td><code>batman-package</code></td><td>&ge;2.4.9</td><td>Mandel-Agol transit model</td></tr>
<tr><td><code>emcee</code></td><td>&ge;3.1.4</td><td>Ensemble MCMC posterior sampling</td></tr>
<tr><td><code>corner</code></td><td>&ge;2.2.2</td><td>MCMC posterior corner plots</td></tr>
<tr><td><code>tensorflow</code></td><td>&ge;2.12</td><td>Dual-view 1D CNN classifier</td></tr>
<tr><td><code>scikit-learn</code></td><td>&ge;1.3</td><td>Metrics, evaluation, preprocessing</td></tr>
<tr><td><code>astropy</code></td><td>&ge;5.0</td><td>Lomb-Scargle, BLS fallback, units</td></tr>
<tr><td><code>astroquery</code></td><td>&ge;0.4.6</td><td>NASA ExoplanetArchive, VizieR, Gaia</td></tr>
<tr><td><code>streamlit</code></td><td>&ge;1.28</td><td>Interactive web dashboard</td></tr>
<tr><td><code>scipy</code></td><td>&ge;1.10</td><td>Optimisation, temperature scaling</td></tr>
<tr><td><code>numpy / pandas / matplotlib</code></td><td>latest</td><td>Numerics, data, visualisation</td></tr>
</table>

<h3>Data Sources</h3>
<table>
<tr><th>Source</th><th>Content</th><th>Used for</th></tr>
<tr><td>NASA MAST (archive.stsci.edu)</td><td>TESS SPOC 2-min LCs + TPFs</td><td>Science dataset</td></tr>
<tr><td>NASA ExoplanetArchive</td><td>Confirmed TESS planet TIC IDs</td><td>Training: class 1</td></tr>
<tr><td>VizieR J/ApJS/258/16 (Prša+ 2022)</td><td>TESS EB catalog</td><td>Training: class 2</td></tr>
<tr><td>ExoFOP-TESS dispositions</td><td>False positives (FP flag)</td><td>Training: class 3</td></tr>
</table>

<hr>
<h2>3 &nbsp; Results</h2>

<h3>3.1 &nbsp; Summary</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Total targets analysed</td><td>{len(results)}</td></tr>
<tr><td>Successfully processed</td><td>{done_n}</td></tr>
<tr><td class="p">Planet transit candidates</td><td>{len(candidates)}</td></tr>
<tr><td class="e">Eclipsing binaries</td><td>{len(ebs)}</td></tr>
<tr><td class="b">Blends / background FP</td><td>{len(blends)}</td></tr>
<tr><td>Stellar variability (starspots)</td><td>{len(var_stars)}</td></tr>
<tr><td class="n">Noise / no signal</td><td>{noise_n}</td></tr>
</table>

<h3>3.2 &nbsp; Planet Transit Candidates</h3>
"""

        if candidates:
            html += """<table>
<tr><th>TIC ID</th><th>Period (d)</th><th>&plusmn;</th>
<th>Depth (%)</th><th>&plusmn; (%)</th>
<th>Duration (h)</th><th>Rp/Rs</th>
<th>SNR</th><th>FAP</th><th>Conf.</th><th>Fit</th></tr>
"""
            for r in sorted(candidates, key=lambda x: -(x.confidence or 0)):
                tls = r.tls_result
                tp  = r.transit_params
                html += f"""<tr>
<td><strong>TIC {r.tic_id}</strong></td>
<td>{tls.period:.5f}</td>
<td>{tls.period_uncertainty:.5f}</td>
<td>{tls.depth*100:.4f}</td>
<td>{'%.4f' % (tp['transit_depth_err']*100) if tp else '—'}</td>
<td>{tls.duration*24:.3f}</td>
<td>{'%.5f' % tp['rp_rs'] if tp else '—'}</td>
<td>{r.snr:.1f}</td>
<td>{f'{r.fap:.2e}' if r.fap else '—'}</td>
<td class="p">{r.confidence*100:.1f}%</td>
<td>{'MCMC' if tp and tp.get('mcmc_used') else 'MAP' if tp else '—'}</td>
</tr>
"""
            html += "</table>"
        else:
            html += "<p>No high-confidence planet candidates in this batch.</p>"

        html += f"""
<h3>3.3 &nbsp; How Uncertainties Are Estimated</h3>
<div class="box">
<p><strong>Period uncertainty</strong> comes directly from TLS, estimated
from the width of the SDE peak in the periodogram.</p>
<p><strong>Transit depth, Rp/Rs, a/Rs, inclination</strong> uncertainties
are the 16th&ndash;84th percentile range of the emcee MCMC posterior
distribution ({'used in this run' if mcmc_used else 'MAP estimates used — enable --mcmc for full posteriors'}).
For non-Gaussian posteriors (common for the impact parameter b),
the asymmetric interval is reported.</p>
<p><strong>False Alarm Probability</strong> is computed by permuting the
detrended flux array {self.config.get('fap_n_bootstrap',100)} times,
re-running TLS on each permutation, and computing the fraction of
noise-only trials that beat the observed SDE. FAP &ge; 1/N always
(never exactly zero from finite sampling).</p>
<p><strong>Confidence</strong> is the calibrated class probability from
the dual-view CNN. Raw softmax values are adjusted via temperature scaling
fit on a held-out validation set, so "80% confidence" reflects ~80%
observed accuracy in that confidence bin.</p>
</div>

<hr>
<p class="meta">
JyotirVega Exoplanet Pipeline v2.0 &nbsp;·&nbsp; Aurixys &nbsp;·&nbsp;
LOGMIEER, Nashik &nbsp;·&nbsp; SPPU<br>
References: Shallue &amp; Vanderburg 2018 &nbsp;|&nbsp;
Hippke &amp; Heller 2019 &nbsp;|&nbsp;
Hippke et al. 2019 &nbsp;|&nbsp;
Kreidberg 2015 &nbsp;|&nbsp;
Foreman-Mackey et al. 2013 &nbsp;|&nbsp;
Prša et al. 2022 &nbsp;|&nbsp;
Guo et al. 2017
</p>
</body></html>"""
        return html

    def _markdown(self, results, candidates):
        lines = [
            "# ISRO PS-7: Exoplanet Detection Report — JyotirVega",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            f"\nTargets analysed: {len(results)}",
            f"Planet candidates: {len(candidates)}",
            "\n## Candidates\n",
            "| TIC | Period (d) | ±(d) | Depth (%) | ±(%) | Dur (h) | SNR | FAP | Conf |",
            "|-----|-----------|------|-----------|------|---------|-----|-----|------|",
        ]
        for r in sorted(candidates, key=lambda x: -(x.confidence or 0)):
            tls = r.tls_result
            tp  = r.transit_params
            lines.append(
                f"| TIC {r.tic_id} "
                f"| {tls.period:.5f} "
                f"| {tls.period_uncertainty:.5f} "
                f"| {tls.depth*100:.4f} "
                f"| {'%.4f' % (tp['transit_depth_err']*100) if tp else '—'} "
                f"| {tls.duration*24:.3f} "
                f"| {r.snr:.1f} "
                f"| {f'{r.fap:.2e}' if r.fap else '—'} "
                f"| {r.confidence*100:.1f}% |"
            )
        return "\n".join(lines)
