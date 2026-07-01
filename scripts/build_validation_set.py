"""
scripts/build_validation_set.py
================================
Builds an INDEPENDENT validation set from real astronomical catalogs.
This is the set that produces the accuracy number in the ISRO report.

CRITICAL: These targets must NOT appear in the training set.

Sources:
  planet_transit   → NASA ExoplanetArchive confirmed TESS planets
  eclipsing_binary → Prša et al. 2022 TESS EB catalog (VizieR J/ApJS/258/16)
  blend            → ExoFOP-TESS TOIs with TFOPWG='FP' disposition
  noise            → ExoFOP-TESS TOIs with TFOPWG='FA' (false alarm)

Usage:
  python scripts/build_validation_set.py --n-per-class 50 --out data/validation
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]  %(message)s')
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Build independent validation set from real TESS catalogs')
    parser.add_argument('--n-per-class', type=int, default=50)
    parser.add_argument('--out',         type=str, default='data/validation')
    args = parser.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    log.info("Building independent validation set:")
    log.info("  Sources: NASA ExoplanetArchive + VizieR J/ApJS/258/16 + ExoFOP")

    rows = []

    # ── Confirmed TESS planets ────────────────────────────────────────────────
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import \
            NasaExoplanetArchive
        log.info("Fetching confirmed TESS planets...")
        tab = NasaExoplanetArchive.query_criteria(
            table  = "pscomppars",
            select = "tic_id,pl_orbper,pl_trandep,pl_trandur",
            where  = "disc_facility like '%TESS%' and tran_flag=1 "
                     "and default_flag=1")
        import pandas as pd
        df = tab.to_pandas().dropna(subset=['tic_id'])
        df['tic_id'] = df['tic_id'].astype(int)
        df['label']  = 'planet_transit'
        df['period'] = df.get('pl_orbper', None)
        sub = df[['tic_id','label','period']].head(args.n_per_class)
        rows.append(sub)
        log.info(f"  {len(sub)} confirmed planet targets")
    except Exception as e:
        log.warning(f"  Planet catalog failed: {e}")

    # ── TESS Eclipsing Binaries ───────────────────────────────────────────────
    try:
        from astroquery.vizier import Vizier
        log.info("Fetching TESS EB catalog (Prša+2022)...")
        Vizier.ROW_LIMIT = args.n_per_class + 50
        result = Vizier.query_constraints("J/ApJS/258/16")
        if result and len(result) > 0:
            import pandas as pd
            df = result[0].to_pandas()
            df = df.rename(columns={'TIC': 'tic_id', 'Per': 'period'})
            df['label'] = 'eclipsing_binary'
            sub = df[['tic_id','label','period']].dropna(
                subset=['tic_id']).head(args.n_per_class)
            sub['tic_id'] = sub['tic_id'].astype(int)
            rows.append(sub)
            log.info(f"  {len(sub)} EB targets")
    except Exception as e:
        log.warning(f"  EB catalog failed: {e}")

    # ── ExoFOP false positives (blends + noise) ───────────────────────────────
    try:
        import requests, pandas as pd
        log.info("Fetching ExoFOP TOI dispositions...")
        url = ("https://exofop.ipac.caltech.edu/tess/download_toi.php"
               "?sort=toi&output=pipe")
        r   = requests.get(url, timeout=45)
        if r.status_code == 200:
            from io import StringIO
            text = r.text
            # Skip comment lines starting with #
            lines = [l for l in text.split('\n') if not l.startswith('#')]
            df    = pd.read_csv(StringIO('\n'.join(lines)), sep='|',
                                skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]

            # FP = blend, FA = noise
            for disp, lbl in [('FP', 'blend'), ('FA', 'noise')]:
                if 'TFOPWG Disposition' in df.columns:
                    sub = df[df['TFOPWG Disposition'].str.strip() == disp]
                elif 'TFOPWG  Disposition' in df.columns:
                    sub = df[df['TFOPWG  Disposition'].str.strip() == disp]
                else:
                    log.warning(f"  Disposition column not found — skipping {lbl}")
                    continue

                if 'TIC ID' in sub.columns:
                    tic_col = 'TIC ID'
                elif 'TIC' in sub.columns:
                    tic_col = 'TIC'
                else:
                    continue

                sub = sub[[tic_col]].rename(columns={tic_col: 'tic_id'}).copy()
                sub['label']  = lbl
                sub['period'] = None
                sub = sub.dropna(subset=['tic_id']).head(args.n_per_class)
                sub['tic_id'] = sub['tic_id'].astype(int)
                rows.append(sub)
                log.info(f"  {len(sub)} {lbl} targets from ExoFOP")

    except Exception as e:
        log.warning(f"  ExoFOP fetch failed: {e}")
        log.warning("  Manual export from https://exofop.ipac.caltech.edu/tess/ "
                    "— filter on TFOPWG Disposition = FP or FA")

    # ── Combine and save ──────────────────────────────────────────────────────
    if not rows:
        log.error("No validation data retrieved. Check internet connection.")
        sys.exit(1)

    import pandas as pd
    combined = pd.concat(rows, ignore_index=True)
    combined = combined.drop_duplicates(subset=['tic_id'])

    out_path = Path(args.out) / 'validation_targets.csv'
    combined.to_csv(out_path, index=False)

    log.info(f"\nSaved {len(combined)} validation targets → {out_path}")
    log.info("\nClass distribution:")
    for lbl, cnt in combined['label'].value_counts().items():
        log.info(f"  {lbl:22s}: {cnt}")

    log.info("\nNext steps:")
    log.info("  1. Run pipeline on these TIC IDs:")
    log.info("     python run_pipeline.py --tics $(python -c \"import pandas as pd; "
             "print(' '.join(str(t) for t in pd.read_csv('data/validation/"
             "validation_targets.csv')['tic_id'].head(50)))\")")
    log.info("  2. Evaluate:")
    log.info("     python scripts/evaluate_validation.py "
             "--predictions outputs/results_table.csv "
             "--ground-truth data/validation/validation_targets.csv")


if __name__ == '__main__':
    main()
