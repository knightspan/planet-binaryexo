# JyotirVega — Complete AI Model Training Guide
### ISRO BAH 2026 · Problem Statement 7

---

## What you are training

A **dual-view 1D CNN** that takes two phase-folded views of a light curve
(global 2001-point + local 201-point) and outputs a 4-class probability:
`noise / planet_transit / eclipsing_binary / blend`.

Architecture follows NASA's AstroNet (Shallue & Vanderburg 2018).

---

## Step 0 — Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Verify GPU (optional, speeds training 10x)
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

## Step 1 — What good training data looks like

Each training sample = (global_view, local_view, features, label).

| Input | Shape | Source |
|---|---|---|
| Global view | (2001,) | Full phase-folded curve, −0.5 to +0.5 |
| Local view | (201,) | Zoomed transit, ±0.075 phase |
| Features | (14,) | Period, depth, SNR, odd/even ratio, etc. |
| Label | int 0-3 | From real catalog or synthetic injection |

**Rules:**
- Aim for ≥200 samples per class, ideally ≥500
- Balance classes or use `class_weight='balanced'` (already done)
- Real data > synthetic for final accuracy
- Validation set must come from a DIFFERENT source than training

---

## Step 2 — Build training data

### Option A: Synthetic (fast, no internet, good for testing the pipeline)

```bash
python train_model.py --synthetic --n-per-class 300 --epochs 0
```

### Option B: Real TESS data (best accuracy, needed for ISRO submission)

```bash
python train_model.py --real --n-per-class 500 --epochs 0
```

This downloads:
- Confirmed TESS planets from NASA ExoplanetArchive → `planet_transit`
- Known EBs from Prša+2022 VizieR catalog → `eclipsing_binary`
- ExoFOP-TESS false positives → `blend`
- Out-of-catalog targets → `noise`

Takes 2-6 hours depending on network. Saves to `data/training/*.npy`.

### Check class balance

```python
import numpy as np
y = np.load('data/training/y_labels.npy')
classes = ['noise', 'planet_transit', 'eclipsing_binary', 'blend']
for i, c in enumerate(classes):
    print(f"{c:20s}: {np.sum(y==i):4d} samples")
```

---

## Step 3 — Training factors to consider

### 3.1 Learning rate

Default `1e-4` is conservative and stable. For larger datasets (1000+/class):

```python
# In pipeline/classifier.py build_model(), swap the optimizer:
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
schedule = CosineDecayRestarts(initial_learning_rate=3e-4,
                                first_decay_steps=500, t_mul=2.0, m_mul=0.9)
optimizer = keras.optimizers.Adam(learning_rate=schedule)
```

### 3.2 Class imbalance

Real TESS sectors are >95% noise. The training script uses `class_weight`
computed from label counts automatically. If the model still over-predicts
noise, manually reduce noise weight by 30-50% in `classifier.py train()`.

### 3.3 Batch size

- GPU available: `batch_size=64`
- CPU only: `batch_size=16` (faster convergence, less memory)

### 3.4 Early stopping (already configured)

```python
keras.callbacks.EarlyStopping(patience=12, restore_best_weights=True,
                              monitor='val_accuracy')
```

500 samples/class → expect convergence in 30-80 epochs.
2000+ samples/class → expect 100-150 epochs.

### 3.5 Overfitting signs

Watch for val_loss rising while train_loss falls:
- Increase dropout: 0.35 → 0.5 in `classifier.py`
- Add L2: `kernel_regularizer=keras.regularizers.l2(1e-4)`
- Get more training data (best fix)

### 3.6 Data augmentation (for minority classes)

```python
def augment(global_view, local_view, y, classes_to_augment=[1,2,3]):
    aug_g, aug_l, aug_y = [], [], []
    for g, l, label in zip(global_view, local_view, y):
        aug_g.append(g); aug_l.append(l); aug_y.append(label)
        if label in classes_to_augment:
            ng = np.clip(g + np.random.normal(0, 0.001, g.shape), 0, 1)
            nl = np.clip(l + np.random.normal(0, 0.001, l.shape), 0, 1)
            aug_g.append(ng); aug_l.append(nl); aug_y.append(label)
    return np.array(aug_g), np.array(aug_l), np.array(aug_y)
```

---

## Step 4 — Run training

```bash
# Quick test
python train_model.py --synthetic --n-per-class 300 --epochs 30 --eval

# Full training
python train_model.py --real --n-per-class 500 --epochs 100 --eval --calibrate
```

### Reading the output

```
Epoch 50: acc=0.82  val_acc=0.78  loss=0.45  val_loss=0.55
```

| val_acc | Quality |
|---|---|
| > 0.75 | Acceptable |
| > 0.82 | Good |
| > 0.90 | Excellent (needs 1000+ samples/class) |

Confusion matrix: watch especially for `planet_transit` ↔ `noise` confusion
(means SNR threshold too low) and `planet_transit` ↔ `eclipsing_binary`
(means odd/even and secondary-eclipse features aren't separating well).

ROC-AUC per class target: > 0.90.

Calibration temperature T:
- T ≈ 1.0 → already well-calibrated
- T > 1.5 → was overconfident, calibration mattered

---

## Step 5 — Independent validation (mandatory for ISRO submission)

The accuracy number in your report MUST come from data the model never trained on.

```bash
# Build independent validation set
python scripts/build_validation_set.py --n-per-class 50

# Run pipeline on validation targets
python run_pipeline.py --tics $(python -c "
import pandas as pd
df = pd.read_csv('data/validation/validation_targets.csv')
print(' '.join(str(int(t)) for t in df['tic_id'].dropna().head(50)))
")

# Compare predictions to ground truth
python scripts/evaluate_validation.py \
    --predictions outputs/results_table.csv \
    --ground-truth data/validation/validation_targets.csv
```

---

## Step 6 — Calibrate confidence

Already run automatically with `--calibrate` flag in training.
Check `models/calibration_temperature.json`:

```json
{
  "temperature": 1.34,
  "reliability_table": [
    {"bin_range": [0.8, 0.9], "n_samples": 45,
     "mean_confidence": 0.84, "observed_accuracy": 0.82}
  ]
}
```

This table goes directly into the report as evidence confidence was checked.

---

## Step 7 — Run inference

```bash
python run_pipeline.py --tics 261136679 388857263
python run_pipeline.py --sector 1 --max-targets 5000
python run_pipeline.py --tics 261136679 --mcmc
python -m streamlit run dashboard/app.py
```

---

## Target numbers for ISRO submission

| Metric | Minimum | Good | Excellent |
|---|---|---|---|
| 4-class validation accuracy | >75% | >82% | >88% |
| Planet transit F1 | >0.70 | >0.80 | >0.90 |
| EB rejection rate | >80% | >90% | >95% |
| Period recovery error | <5% | <1% | <0.1% |
| Depth recovery error | <20% | <10% | <5% |
| Bootstrap FAP at SNR=9 | <0.01 | <0.001 | <1e-4 |

---

## Common mistakes

1. Reporting train-split accuracy instead of independent validation accuracy
2. Not calibrating — raw softmax overconfident at high values
3. Single-sector training only — TESS systematics vary by sector
4. Equal class weights with 10:1 imbalance
5. Skipping the variability screen — inflates EB false-positive rate
6. Single-point parameter estimates without MCMC posteriors
7. Not saving the calibration temperature for the report

---

## File structure after training

```
models/
  classifier.h5
  calibration_temperature.json
  calibration_reliability.json

data/
  training/{X_global,X_local,X_features,y_labels}.npy
  validation/validation_targets.csv

outputs/
  confusion_matrix.png
  training_history.png
  pipeline.log
```
