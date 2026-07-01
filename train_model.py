"""
train_model.py
==============
Complete training script for the JyotirVega dual-view CNN classifier.
ISRO BAH 2026 — Problem Statement 7

Trains on REAL TESS data (downloaded from MAST) or synthetic data.
Includes:
  - Class-balanced training with class_weight
  - Early stopping + learning rate reduction
  - Full evaluation: confusion matrix, ROC-AUC, classification report
  - Temperature-scaling confidence calibration
  - Independent validation set accuracy (the number that goes in the report)

Usage:
  # Quick test — synthetic data, no internet (3-5 min)
  python train_model.py --synthetic --epochs 30 --n-per-class 300

  # Best accuracy — real TESS data (takes 1-3 hours)
  python train_model.py --real --epochs 100 --n-per-class 500

  # With full evaluation + calibration
  python train_model.py --synthetic --epochs 50 --eval --calibrate
"""

import argparse
import numpy as np
import logging
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s]  %(message)s')
log = logging.getLogger(__name__)

CLASSES = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']

DEFAULT_CONFIG = {
    'min_snr':               7.0,
    'min_transit_count':     3,
    'period_min':            0.5,
    'period_max':            13.0,
    'sigma_clip':            5.0,
    'wotan_window':          0.75,
    'wotan_method':          'biweight',
    'run_centroid':          False,
    'run_variability_screen': False,
    'run_blend_crosscheck':  False,
    'run_fap_bootstrap':     False,
    'run_multiplanet_search':False,
    'output_dir':            'outputs',
    'model_path':            'models/classifier.h5',
    'cache_dir':             'data/cache',
    'random_seed':           42,
    'rotation_fap_threshold':  1e-3,
    'spot_amp_ratio_threshold':5.0,
    'spot_duty_cycle_threshold':0.25,
    'rotation_period_min':   0.1,
    'rotation_period_max':   30.0,
    'contamination_ratio_threshold': 0.10,
    'blend_search_radius_px':2.0,
    'blend_max_neighbor_mag_diff':4.0,
    'fap_n_bootstrap':       100,
    'max_planets_per_target':1,
    'multiplanet_sde_threshold':7.0,
}


def main():
    parser = argparse.ArgumentParser(
        description='Train JyotirVega dual-view CNN classifier')
    parser.add_argument('--synthetic',     action='store_true',
                        help='Train on synthetic data (no internet, fast)')
    parser.add_argument('--real',          action='store_true',
                        help='Download real TESS data and train')
    parser.add_argument('--eval',          action='store_true',
                        help='Full evaluation after training')
    parser.add_argument('--calibrate',     action='store_true', default=True,
                        help='Temperature-scale calibration on val split')
    parser.add_argument('--epochs',        type=int, default=50)
    parser.add_argument('--n-per-class',   type=int, default=300)
    parser.add_argument('--batch-size',    type=int, default=32)
    parser.add_argument('--load-existing', action='store_true',
                        help='Load saved dataset instead of regenerating')
    args = parser.parse_args()

    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    Path('data/training').mkdir(parents=True, exist_ok=True)

    # ── Build or load dataset ─────────────────────────────────────────────────
    from pipeline.training_data_builder import TrainingDataBuilder
    builder = TrainingDataBuilder(DEFAULT_CONFIG, output_dir='data/training')

    if args.load_existing:
        log.info("Loading existing dataset...")
        try:
            Xg, Xl, Xf, y = builder.load_dataset(synthetic=args.synthetic)
        except FileNotFoundError:
            log.error("No saved dataset found. Run without --load-existing.")
            sys.exit(1)

    elif args.synthetic or not args.real:
        log.info(f"Building synthetic dataset ({args.n_per_class}/class)...")
        Xg, Xl, Xf, y = builder.build_synthetic_dataset(
            n_per_class=args.n_per_class)

    else:
        log.info("Fetching real TIC lists from NASA ExoplanetArchive / VizieR...")
        tic_lists = builder.fetch_tic_lists(n_per_class=args.n_per_class)
        log.info("Downloading and processing real TESS light curves...")
        Xg, Xl, Xf, y = builder.build_dataset(tic_lists)

    log.info(f"\nDataset: global={Xg.shape}  local={Xl.shape}  "
             f"features={Xf.shape}  labels={y.shape}")
    for i, c in enumerate(CLASSES):
        log.info(f"  Class {i} ({c}): {int(np.sum(y==i))} samples")

    if len(y) < 20:
        log.error("Insufficient samples. Check installation of TLS/wotan.")
        sys.exit(1)

    # ── Train ─────────────────────────────────────────────────────────────────
    from pipeline.classifier import ExoplanetClassifier
    clf = ExoplanetClassifier(DEFAULT_CONFIG)

    log.info(f"\nTraining dual-view CNN for up to {args.epochs} epochs...")
    history = clf.train(
        X_global   = Xg,
        X_local    = Xl,
        X_features = Xf,
        y_labels   = y,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        val_split  = 0.20,
        save_path  = 'models/classifier.h5',
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    if args.eval and history is not None and clf.model is not None:
        _evaluate(clf, Xg, Xl, Xf, y, calibrate=args.calibrate)

    log.info("\nTraining complete.")
    log.info("Model saved: models/classifier.h5")
    log.info("\nIMPORTANT: For the ISRO submission, report accuracy from the")
    log.info("INDEPENDENT validation set, not this training-distribution split.")
    log.info("Run: python scripts/build_validation_set.py")


# ─────────────────────────────────────────────────────────────────────────────
def _evaluate(clf, Xg, Xl, Xf, y, calibrate=True):
    try:
        from sklearn.metrics import (classification_report, confusion_matrix,
                                     roc_auc_score)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import label_binarize
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        log.warning(f"Eval deps missing: {e}")
        return

    log.info("\n" + "="*50)
    log.info("EVALUATION (train-distribution split — see note above)")
    log.info("="*50)

    _, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)

    Xg_t = Xg[test_idx].reshape(-1, 2001, 1)
    Xl_t = Xl[test_idx].reshape(-1,  201, 1)
    y_t  = y[test_idx]

    probs = clf.model.predict(
        {'global_view': Xg_t, 'local_view': Xl_t}, verbose=0)
    preds = np.argmax(probs, axis=1)

    log.info("\nClassification Report:")
    log.info(classification_report(y_t, preds, target_names=CLASSES, digits=3))

    cm = confusion_matrix(y_t, preds)
    log.info(f"Confusion Matrix:\n{cm}")

    y_bin = label_binarize(y_t, classes=[0, 1, 2, 3])
    try:
        auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
        log.info(f"Macro-average ROC-AUC: {auc:.4f}")
    except Exception as e:
        log.warning(f"ROC-AUC failed: {e}")

    # Confusion matrix plot
    _plot_cm(cm, 'outputs/confusion_matrix.png')

    # Training history
    if hasattr(clf, '_last_history') and clf._last_history:
        _plot_history(clf._last_history, 'outputs/training_history.png')

    # Calibration
    if calibrate:
        _calibrate(clf, Xg_t, Xl_t, y_t)


def _calibrate(clf, Xg_t, Xl_t, y_t):
    from pipeline.calibration import ConfidenceCalibrator
    try:
        probs        = clf.model.predict(
            {'global_view': Xg_t, 'local_view': Xl_t}, verbose=0)
        logits_approx = np.log(np.clip(probs, 1e-12, 1.0))

        cal   = ConfidenceCalibrator()
        T     = cal.fit(logits_approx, y_t)
        table = cal.reliability_table(logits_approx, y_t)

        log.info(f"\nCalibration temperature T={T:.3f}")
        log.info("Reliability table (for report):")
        for row in table:
            log.info(f"  bin {row['bin_range']}: n={row['n_samples']:>3}  "
                     f"mean_conf={row['mean_confidence']:.3f}  "
                     f"obs_acc={row['observed_accuracy']:.3f}")

        cal.save('models/calibration_temperature.json')
        # Also save reliability table for report
        with open('models/calibration_reliability.json', 'w') as f:
            json.dump({'temperature': T, 'reliability_table': table}, f, indent=2)
        log.info("Calibration saved: models/calibration_temperature.json")

    except Exception as e:
        log.warning(f"Calibration failed (non-fatal): {e}")


def _plot_cm(cm, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PAL = {'bg': '#0A0A0F', 'panel': '#12121A', 'text': '#E8E8F0',
           'accent': '#C9A84C'}
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(PAL['bg'])
    ax.set_facecolor(PAL['panel'])
    cm_n = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im   = ax.imshow(cm_n, cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([c.replace('_','\n') for c in CLASSES],
                       color=PAL['text'], fontsize=9)
    ax.set_yticklabels(CLASSES, color=PAL['text'], fontsize=9)
    ax.set_xlabel('Predicted', color=PAL['text'])
    ax.set_ylabel('True',      color=PAL['text'])
    ax.set_title('Normalised Confusion Matrix', color=PAL['accent'], fontsize=12)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i,j]}\n({cm_n[i,j]*100:.0f}%)",
                    ha='center', va='center', fontsize=8,
                    color='white' if cm_n[i,j] > 0.5 else '#888790')
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close()
    log.info(f"Confusion matrix saved: {path}")


def _plot_history(history, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PAL = {'bg': '#0A0A0F', 'panel': '#12121A', 'text': '#E8E8F0',
           'accent': '#C9A84C', 'blue': '#4A90D9'}
    h    = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor(PAL['bg'])
    for ax in axes:
        ax.set_facecolor(PAL['panel'])
        ax.tick_params(colors=PAL['text'])
        for sp in ax.spines.values():
            sp.set_edgecolor('#222230')
    axes[0].plot(h.get('accuracy', []),     color=PAL['blue'],   label='Train')
    axes[0].plot(h.get('val_accuracy', []), color=PAL['accent'], label='Val')
    axes[0].set_title('Accuracy',  color=PAL['accent'])
    axes[0].set_xlabel('Epoch',    color=PAL['text'])
    axes[0].set_ylabel('Accuracy', color=PAL['text'])
    axes[0].legend(); axes[0].grid(True, alpha=0.2)
    axes[1].plot(h.get('loss', []),     color=PAL['blue'],   label='Train')
    axes[1].plot(h.get('val_loss', []), color=PAL['accent'], label='Val')
    axes[1].set_title('Loss',  color=PAL['accent'])
    axes[1].set_xlabel('Epoch', color=PAL['text'])
    axes[1].set_ylabel('Loss',  color=PAL['text'])
    axes[1].legend(); axes[1].grid(True, alpha=0.2)
    plt.suptitle('DualViewCNN Training History', color=PAL['accent'], fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=PAL['bg'])
    plt.close()
    log.info(f"Training history saved: {path}")


if __name__ == '__main__':
    main()
