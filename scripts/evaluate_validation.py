"""
scripts/evaluate_validation.py
================================
Compare pipeline predictions to ground-truth labels from the independent
validation set. Produces the numbers that go in the ISRO PS-7 report.

Usage:
  python scripts/evaluate_validation.py \
      --predictions outputs/results_table.csv \
      --ground-truth data/validation/validation_targets.csv
"""

import argparse
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]  %(message)s')
log = logging.getLogger(__name__)

CLASSES = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate pipeline against independent validation set')
    parser.add_argument('--predictions',  required=True,
                        help='Pipeline CSV output (results_table.csv)')
    parser.add_argument('--ground-truth', required=True,
                        help='Validation CSV with tic_id + label columns')
    parser.add_argument('--out',          default='outputs/validation_report.txt')
    args = parser.parse_args()

    pred  = pd.read_csv(args.predictions)
    truth = pd.read_csv(args.ground_truth)

    # Normalise column names
    for col_map in [('TIC', 'tic_id'), ('Classification', 'classification'),
                    ('Class', 'classification')]:
        if col_map[0] in pred.columns:
            pred = pred.rename(columns={col_map[0]: col_map[1]})

    pred['tic_id']  = pred['tic_id'].astype(int)
    truth['tic_id'] = truth['tic_id'].astype(int)

    merged = pred.merge(truth[['tic_id', 'label']], on='tic_id', how='inner')
    merged = merged.dropna(subset=['classification', 'label'])

    n = len(merged)
    if n == 0:
        log.error("No matching TIC IDs between predictions and ground truth.")
        log.error(f"Predictions TICs: {pred['tic_id'].head(5).tolist()}")
        log.error(f"Truth TICs:       {truth['tic_id'].head(5).tolist()}")
        sys.exit(1)

    log.info(f"Matched {n} targets for evaluation")

    from sklearn.metrics import classification_report, confusion_matrix

    y_true = merged['label'].values
    y_pred = merged['classification'].values

    report  = classification_report(y_true, y_pred, digits=3, zero_division=0)
    cm      = confusion_matrix(y_true, y_pred, labels=CLASSES)
    acc     = (y_true == y_pred).mean() * 100

    lines = [
        "=" * 65,
        "JyotirVega — INDEPENDENT VALIDATION RESULTS",
        "ISRO BAH 2026 · Problem Statement 7 · Team JyotirVega",
        "=" * 65,
        f"\nTotal targets evaluated:  {n}",
        f"Overall accuracy:          {acc:.1f}%",
        f"\nClassification Report:\n{report}",
        f"Confusion Matrix (rows=true, cols=predicted):",
        f"Labels: {CLASSES}",
        str(cm),
        "\nPer-class breakdown:",
    ]

    for cls in CLASSES:
        mask = (y_true == cls)
        if mask.sum() == 0:
            continue
        cls_acc = (y_pred[mask] == cls).mean() * 100
        lines.append(f"  {cls:22s}: {cls_acc:.1f}%  "
                     f"({int(mask.sum())} targets)")

    # Parameter accuracy for planet candidates
    planet_mask = ((merged['label'] == 'planet_transit') &
                   (merged['classification'] == 'planet_transit'))
    if planet_mask.sum() > 0:
        lines.append(f"\nCorrectly classified planet transits: "
                     f"{int(planet_mask.sum())}")

        for col, name in [('fit_period',   'Period (d)'),
                          ('fit_depth_pct','Depth (%)'),
                          ('fit_duration_h','Duration (h)')]:
            if col in merged.columns and f'{col}_truth' not in merged.columns:
                vals = merged.loc[planet_mask, col].dropna()
                if len(vals) > 0:
                    lines.append(f"  {name}: mean={vals.mean():.4f}  "
                                 f"std={vals.std():.4f}  "
                                 f"({len(vals)} fitted)")

    lines += [
        "\n" + "=" * 65,
        "CONFIDENCE CALIBRATION NOTE:",
        "Confidence values are temperature-scaled (post-hoc calibration).",
        "See models/calibration_temperature.json for reliability table.",
        "A well-calibrated model has observed_accuracy ≈ mean_confidence",
        "in each bin of the reliability table.",
        "=" * 65,
    ]

    output = "\n".join(lines)
    print(output)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(output)
    log.info(f"Validation report saved: {args.out}")

    # Save per-target comparison CSV
    detail_path = args.out.replace('.txt', '_detail.csv')
    merged[['tic_id', 'label', 'classification']].to_csv(
        detail_path, index=False)
    log.info(f"Detail CSV saved: {detail_path}")


if __name__ == '__main__':
    main()
