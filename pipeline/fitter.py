"""
pipeline/fitter.py
==================
Transit parameter fitting using batman (Mandel-Agol model) + emcee MCMC.

Two stages:
  1. MAP estimate via Nelder-Mead optimisation (fast, ~seconds)
     → good starting point for MCMC
  2. emcee MCMC posterior sampling (slower, ~minutes)
     → full posterior distributions → asymmetric uncertainties
     → 16th–84th percentile = 1σ credible intervals

Outputs per transit candidate:
  period          ± period_err        (from TLS)
  t0              ± t0_err            (from MCMC)
  transit_depth   ± transit_depth_err (= Rp/Rs² ± propagated)
  transit_duration ± transit_duration_err
  rp_rs           ± rp_rs_err
  a_rs            ± a_rs_err
  inclination     ± inclination_err
  impact_parameter

Fixed parameters: eccentricity=0 (circular), limb darkening u=[0.40, 0.26]
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


class TransitParameterFitter:

    def __init__(self, config: dict):
        self.mcmc_walkers = config.get('mcmc_walkers', 32)
        self.mcmc_steps   = config.get('mcmc_steps',   3000)
        self.mcmc_burnin  = config.get('mcmc_burnin',  500)

    def fit(self, time, flux, flux_err, tls_result, run_mcmc=True):
        """
        Returns (transit_params dict, mcmc_samples array or None).
        transit_params always returned even if MCMC is skipped.
        """
        p0 = self._initial_guess(tls_result)
        log.info(f"  Initial guess: rp={p0['rp']:.4f}  "
                 f"a={p0['a']:.1f}  inc={p0['inc']:.1f}°")

        # Stage 1: MAP
        map_params = self._map_fit(time, flux, flux_err, p0)

        # Stage 2: MCMC
        samples    = None
        mcmc_stats = None
        if run_mcmc:
            try:
                samples, mcmc_stats = self._run_mcmc(
                    time, flux, flux_err, map_params)
            except Exception as e:
                log.warning(f"  MCMC failed ({e}) — using MAP")

        return self._package(tls_result, map_params, mcmc_stats), samples

    # ─────────────────────────────────────────────────────────────────────────
    def _initial_guess(self, r):
        rp = float(np.sqrt(max(r.depth, 1e-8)))
        a  = float(np.clip(np.pi * r.period / max(r.duration, 0.01), 2, 200))
        return {'t0': float(r.T0), 'period': float(r.period),
                'rp': rp, 'a': a, 'inc': 88.0, 'u1': 0.40, 'u2': 0.26}

    def _map_fit(self, time, flux, flux_err, p0):
        try:
            import batman
            from scipy.optimize import minimize

            bp = batman.TransitParams()
            bp.per        = p0['period']
            bp.ecc        = 0.0
            bp.w          = 90.0
            bp.limb_dark  = "quadratic"
            bp.u          = [p0['u1'], p0['u2']]

            def loss(theta):
                t0, rp, a, inc = theta
                if rp < 0.001 or rp > 0.5: return 1e12
                if a  < 1.5   or a  > 200: return 1e12
                if inc < 60   or inc > 91:  return 1e12
                bp.t0  = t0; bp.rp = rp; bp.a = a; bp.inc = inc
                m      = batman.TransitModel(bp, time)
                return float(np.sum(((flux - m.light_curve(bp)) / flux_err)**2))

            res = minimize(
                loss, [p0['t0'], p0['rp'], p0['a'], p0['inc']],
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8})

            t0f, rpf, af, incf = res.x
            log.info(f"  MAP fit: rp={abs(rpf):.5f}  a={abs(af):.2f}  "
                     f"inc={incf:.2f}°  χ²={res.fun:.1f}")
            return {'t0': float(t0f), 'period': p0['period'],
                    'rp': float(abs(rpf)), 'a': float(abs(af)),
                    'inc': float(incf), 'u1': p0['u1'], 'u2': p0['u2']}

        except ImportError:
            log.warning("  batman not installed — using TLS parameters as MAP")
            return p0
        except Exception as e:
            log.warning(f"  MAP failed ({e}) — using initial guess")
            return p0

    def _run_mcmc(self, time, flux, flux_err, p0):
        import emcee, batman

        period = p0['period']
        u1, u2 = p0['u1'], p0['u2']

        bp = batman.TransitParams()
        bp.per = period; bp.ecc = 0.0; bp.w = 90.0
        bp.limb_dark = "quadratic"; bp.u = [u1, u2]

        def log_prior(theta):
            t0, rp, a, inc = theta
            if not (p0['t0'] - 0.5 < t0 < p0['t0'] + 0.5): return -np.inf
            if not (0.001 < rp < 0.5):                       return -np.inf
            if not (1.5 < a < 200):                           return -np.inf
            if not (60 < inc < 91):                           return -np.inf
            return 0.0

        def log_like(theta):
            t0, rp, a, inc = theta
            bp.t0 = t0; bp.rp = rp; bp.a = a; bp.inc = inc
            m  = batman.TransitModel(bp, time)
            mf = m.light_curve(bp)
            return -0.5 * float(np.sum(((flux - mf) / flux_err)**2))

        def log_prob(theta):
            lp = log_prior(theta)
            return -np.inf if not np.isfinite(lp) else lp + log_like(theta)

        ndim     = 4
        nwalkers = max(self.mcmc_walkers, 2 * ndim)
        p_init   = np.array([p0['t0'], p0['rp'], p0['a'], p0['inc']])
        pos      = p_init + np.array([1e-4, 1e-4, 0.5, 0.1]) * \
                   np.random.randn(nwalkers, ndim)

        log.info(f"  MCMC: {nwalkers} walkers × {self.mcmc_steps} steps")
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
        sampler.run_mcmc(pos, self.mcmc_steps, progress=False)

        samples = sampler.get_chain(discard=self.mcmc_burnin, flat=True)
        log.info(f"  MCMC done: {len(samples)} posterior samples")

        param_stats = {}
        for i, name in enumerate(['t0', 'rp', 'a', 'inc']):
            q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
            param_stats[name] = {'median': q50, 'err_lo': q50-q16,
                                  'err_hi': q84-q50}
            log.info(f"    {name}: {q50:.5f} +{q84-q50:.5f} -{q50-q16:.5f}")

        return samples, param_stats

    def _package(self, r, mp, mcmc):
        def gv(key, fallback):
            if mcmc and key in mcmc:
                s = mcmc[key]
                return float(s['median']), max(float(s['err_lo']),
                                               float(s['err_hi']))
            return float(mp.get(key, fallback)), float(fallback) * 0.05

        rp,  rp_err  = gv('rp',  mp.get('rp',  0.05))
        a,   a_err   = gv('a',   mp.get('a',   15.0))
        t0,  t0_err  = gv('t0',  mp.get('t0',  r.T0))
        inc, inc_err = gv('inc', mp.get('inc', 88.0))

        depth     = rp ** 2
        depth_err = 2 * rp * rp_err
        b         = a * np.cos(np.deg2rad(inc))

        try:
            arg = np.sqrt(max((1 + rp)**2 - b**2, 0)) / a
            dur = float(r.period / np.pi * np.arcsin(np.clip(arg, -1, 1)))
        except Exception:
            dur = float(r.duration)

        return {
            'period':               float(r.period),
            'period_err':           float(r.period_uncertainty),
            't0':                   float(t0),
            't0_err':               float(t0_err),
            'rp_rs':                float(rp),
            'rp_rs_err':            float(rp_err),
            'a_rs':                 float(a),
            'a_rs_err':             float(a_err),
            'inclination':          float(inc),
            'inclination_err':      float(inc_err),
            'transit_depth':        float(depth),
            'transit_depth_err':    float(depth_err),
            'transit_duration':     float(dur),
            'transit_duration_err': float(r.duration * 0.05),
            'impact_parameter':     float(b),
            'mcmc_used':            mcmc is not None,
        }
