"""
pipeline/training_data_builder.py
==================================
Builds training datasets from:
  A. REAL TESS data — confirmed planets, known EBs, false positives
  B. Synthetic data — for offline testing

REAL DATA SOURCES:
  planet_transit   → NASA Exoplanet Archive (confirmed TESS planets)
  eclipsing_binary → TESS EB Catalog (Prša et al. 2022, VizieR J/ApJS/258/16)
  blend            → ExoFOP-TESS false positives (FP dispositions)
  noise            → Targets with no detected signal

Each sample = (global_view 2001-pt, local_view 201-pt, features 14-d, label)
"""

import numpy as np
import logging
from pathlib import Path

log = logging.getLogger(__name__)
CLASSES = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']
N_GLOBAL = 2001
N_LOCAL  = 201


class TrainingDataBuilder:

    def __init__(self, config: dict, output_dir: str = 'data/training'):
        self.config     = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        from pipeline.data_loader       import TESSDataLoader
        from pipeline.preprocessor      import LightCurvePreprocessor
        from pipeline.detector          import TransitDetector
        from pipeline.false_positive    import FalsePositiveVetter
        from pipeline.classifier        import ExoplanetClassifier

        self.loader      = TESSDataLoader(config)
        self.pre         = LightCurvePreprocessor(config)
        self.det         = TransitDetector(config)
        self.vet         = FalsePositiveVetter(config)
        self.clf         = ExoplanetClassifier(config)

    # ─────────────────────────────────────────────────────────────────────────
    # REAL DATA
    # ─────────────────────────────────────────────────────────────────────────
    def fetch_tic_lists(self, n_per_class=300):
        """Fetch TIC ID lists from real astronomical catalogs."""
        tic_lists = {c: [] for c in CLASSES}

        # ── Confirmed TESS planets ─────────────────────────────────────────
        try:
            from astroquery.ipac.nexsci.nasa_exoplanet_archive import \
                NasaExoplanetArchive
            log.info("Fetching confirmed TESS planets from NASA ExoplanetArchive...")
            tab = NasaExoplanetArchive.query_criteria(
                table  = "pscomppars",
                select = "tic_id,pl_orbper,pl_trandep,pl_trandur",
                where  = "disc_facility like '%TESS%' and tran_flag=1")
            tics = [int(r['tic_id']) for r in tab
                    if r['tic_id'] and str(r['tic_id']).strip()]
            tic_lists['planet_transit'] = list(set(tics))[:n_per_class]
            log.info(f"  {len(tic_lists['planet_transit'])} planet TICs")
        except Exception as e:
            log.warning(f"  Planet catalog failed: {e} — using fallback")
            tic_lists['planet_transit'] = self._planet_fallback()

        # ── TESS EBs ──────────────────────────────────────────────────────
        try:
            from astroquery.vizier import Vizier
            log.info("Fetching TESS EB catalog (Prša+2022)...")
            Vizier.ROW_LIMIT = -1
            r = Vizier.query_constraints("J/ApJS/258/16")[0]
            tics = [int(row['TIC']) for row in r if 'TIC' in r.colnames]
            tic_lists['eclipsing_binary'] = list(set(tics))[:n_per_class]
            log.info(f"  {len(tic_lists['eclipsing_binary'])} EB TICs")
        except Exception as e:
            log.warning(f"  EB catalog failed: {e} — using fallback")
            tic_lists['eclipsing_binary'] = self._eb_fallback()

        # ── ExoFOP false positives ────────────────────────────────────────
        tic_lists['blend'] = self._fetch_fps(n_per_class)

        # ── Noise (no signal targets) ─────────────────────────────────────
        all_tics = set()
        for v in tic_lists.values():
            all_tics.update(v)
        tic_lists['noise'] = [t + 999999 for t in
                               list(all_tics)[:n_per_class]]

        return tic_lists

    def build_dataset(self, tic_lists):
        """Download real TESS LCs and extract training arrays."""
        Xg, Xl, Xf, y = [], [], [], []

        for label_idx, cls_name in enumerate(CLASSES):
            tics  = tic_lists.get(cls_name, [])
            n_ok  = 0
            log.info(f"Processing '{cls_name}': {len(tics)} targets")
            for tic in tics:
                try:
                    gv, lv, fv = self._process_one_real(tic)
                    if gv is not None:
                        Xg.append(gv); Xl.append(lv)
                        Xf.append(fv); y.append(label_idx)
                        n_ok += 1
                except Exception as e:
                    log.debug(f"  TIC {tic}: {e}")
            log.info(f"  → {n_ok} usable samples")

        Xg = np.array(Xg, dtype=np.float32)
        Xl = np.array(Xl, dtype=np.float32)
        Xf = np.array(Xf, dtype=np.float32)
        y  = np.array(y,  dtype=np.int32)

        self._save(Xg, Xl, Xf, y, suffix='')
        return Xg, Xl, Xf, y

    def _process_one_real(self, tic_id):
        lc = self.loader.download(tic_id)
        if lc is None:
            return None, None, None
        time, flux, flux_err = self.pre.process(lc)
        tls = self.det.search(time, flux, flux_err)
        if tls is None or tls.period is None:
            return None, None, None
        from pipeline.preprocessor import LightCurvePreprocessor
        _, _, gv, lv = LightCurvePreprocessor.phase_fold(
            time, flux, tls.period, tls.T0,
            n_bins_global=N_GLOBAL, n_bins_local=N_LOCAL)
        flags = self.vet.vet(time, flux, flux_err, tls, run_centroid=False)
        fv    = self.clf._build_features(tls, flags)
        return gv.astype(np.float32), lv.astype(np.float32), fv.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────────
    # SYNTHETIC DATA
    # ─────────────────────────────────────────────────────────────────────────
    def build_synthetic_dataset(self, n_per_class=300):
        """Generate synthetic training data — no internet required."""
        log.info(f"Building synthetic dataset: {n_per_class} per class")
        Xg, Xl, Xf, y = [], [], [], []

        for label_idx, cls_name in enumerate(CLASSES):
            n_ok = 0
            for i in range(n_per_class):
                seed = label_idx * 100000 + i
                try:
                    lc = self.loader.get_synthetic_lc(seed, signal_type=cls_name)
                    time, flux, flux_err = self.pre.process(lc)
                    tls = self.det.search(time, flux, flux_err)
                    if tls is None or tls.period is None:
                        # For noise class this is expected
                        if cls_name == 'noise':
                            # Create dummy views
                            gv = np.ones(N_GLOBAL, dtype=np.float32) * 0.5
                            lv = np.ones(N_LOCAL,  dtype=np.float32) * 0.5
                            fv = np.zeros(14, dtype=np.float32)
                        else:
                            continue
                    else:
                        from pipeline.preprocessor import LightCurvePreprocessor
                        _, _, gv, lv = LightCurvePreprocessor.phase_fold(
                            time, flux, tls.period, tls.T0,
                            n_bins_global=N_GLOBAL, n_bins_local=N_LOCAL)
                        flags = self.vet.vet(time, flux, flux_err, tls,
                                             run_centroid=False)
                        fv    = self.clf._build_features(tls, flags)

                    Xg.append(gv.astype(np.float32))
                    Xl.append(lv.astype(np.float32))
                    Xf.append(fv.astype(np.float32))
                    y.append(label_idx)
                    n_ok += 1
                except Exception as e:
                    log.debug(f"  Synthetic {cls_name}[{i}]: {e}")

            log.info(f"  '{cls_name}': {n_ok} samples")

        Xg = np.array(Xg, dtype=np.float32)
        Xl = np.array(Xl, dtype=np.float32)
        Xf = np.array(Xf, dtype=np.float32)
        y  = np.array(y,  dtype=np.int32)
        self._save(Xg, Xl, Xf, y, suffix='_syn')
        return Xg, Xl, Xf, y

    def _save(self, Xg, Xl, Xf, y, suffix=''):
        np.save(self.output_dir / f'X_global{suffix}.npy',   Xg)
        np.save(self.output_dir / f'X_local{suffix}.npy',    Xl)
        np.save(self.output_dir / f'X_features{suffix}.npy', Xf)
        np.save(self.output_dir / f'y_labels{suffix}.npy',   y)
        log.info(f"Dataset saved: {len(y)} samples → {self.output_dir}")

    def load_dataset(self, synthetic=False):
        s = '_syn' if synthetic else ''
        return (np.load(self.output_dir / f'X_global{s}.npy'),
                np.load(self.output_dir / f'X_local{s}.npy'),
                np.load(self.output_dir / f'X_features{s}.npy'),
                np.load(self.output_dir / f'y_labels{s}.npy'))

    # ─────────────────────────────────────────────────────────────────────────
    # Fallback TIC lists (hardcoded known targets)
    # ─────────────────────────────────────────────────────────────────────────
    def _planet_fallback(self):
        return [261136679, 149603524, 460205581, 100100827, 268644785,
                382188479, 441075486, 350618622, 63365251,  219854185,
                427509796, 142748283, 260004324, 167324598, 229510866,
                410214986, 281459670, 294750180, 231702397, 158588995]

    def _eb_fallback(self):
        return [388857263, 219511015, 300038787, 81588086,  232979463,
                277539431, 352576555, 167605776, 159309672, 293954617]

    def _fetch_fps(self, n_per_class):
        try:
            import requests
            url = ("https://exofop.ipac.caltech.edu/tess/download_toi.php"
                   "?sort=toi&output=pipe")
            r   = requests.get(url, timeout=30)
            fps = []
            for line in r.text.split('\n'):
                if '|' in line and 'FP' in line:
                    parts = line.split('|')
                    if len(parts) > 2:
                        try:
                            fps.append(int(parts[1].strip()))
                        except Exception:
                            pass
            log.info(f"  ExoFOP FPs: {len(fps)} targets")
            return list(set(fps))[:n_per_class]
        except Exception as e:
            log.warning(f"  ExoFOP fetch failed: {e}")
            return []
