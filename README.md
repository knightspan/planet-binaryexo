# 🪐 JyotirVega Exoplanet Detection Pipeline
### ISRO Bharatiya Antariksh Hackathon 2026 — Problem Statement 7
**Team JyotirVega | Aurixys | LOGMIEER, Nashik | SPPU**

---

## What this is

A complete, 10-stage AI pipeline that downloads real TESS light curves from
NASA MAST, screens for stellar variability, detrends with Wotan biweight,
searches for transits with Transit Least Squares, vets false positives with
5 independent physics tests, classifies signals with a dual-view CNN, and
fits transit parameters with batman + emcee MCMC.

Every PS-7 objective is addressed:
| PS-7 Requirement | Implementation |
|---|---|
| Identify periodic dips | TLS limb-darkened transit search |
| Classify into transit/eclipse/blend/other | Dual-view CNN + rule ensemble + variability screen |
| Apply to science datasets | `--sector` batch mode, `--tics` mode |
| SNR / significance | Folded SNR + TLS SDE + bootstrap FAP |
| Period, depth, duration estimates | batman MAP → emcee MCMC posteriors |
| Confidence level | Temperature-scaled calibrated probability |
| 3-page report | Auto-generated HTML + Markdown |
| Visualization | 6-panel diagnostic + corner plots + population |

---

## Windows Quick Start

```
1. Double-click setup_windows.bat    ← installs everything
2. Double-click run_demo.bat         ← runs pipeline, no internet needed
3. Double-click run_dashboard.bat    ← opens web dashboard at localhost:8501
```

---

## Linux / Mac Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Demo (no internet)
python run_pipeline.py --demo

# Dashboard
python -m streamlit run dashboard/app.py

# Train CNN
python train_model.py --synthetic --epochs 50 --n-per-class 300 --eval
```

---

## All commands

```bash
# Synthetic demo (no internet, fast)
python run_pipeline.py --demo

# Demo with MCMC (slower, full posterior uncertainties)
python run_pipeline.py --demo --mcmc

# Real TESS targets (needs internet)
python run_pipeline.py --tics 261136679 388857263 100100827

# Full sector batch (20-30k targets)
python run_pipeline.py --sector 1 --max-targets 100

# Known targets demo
python run_pipeline.py --real-demo

# Train CNN — synthetic (fast, ~5 min)
python train_model.py --synthetic --epochs 50 --n-per-class 300 --eval --calibrate

# Train CNN — real TESS data (best accuracy, takes hours)
python train_model.py --real --epochs 100 --n-per-class 500 --eval --calibrate

# Build independent validation set
python scripts/build_validation_set.py --n-per-class 50

# Evaluate against ground truth
python scripts/evaluate_validation.py \
    --predictions outputs/results_table.csv \
    --ground-truth data/validation/validation_targets.csv

# Interactive dashboard
python -m streamlit run dashboard/app.py
```

---

## Pipeline stages

| Stage | Module | What it does |
|---|---|---|
| 1 | `data_loader.py` | Download real TESS SPOC 2-min LC from MAST |
| 2 | `stellar_variability.py` | Lomb-Scargle starspot pre-screen |
| 3 | `preprocessor.py` | Wotan biweight flatten + sigma clip |
| 4 | `detector.py` + `multiplanet.py` | TLS search + iterative multi-planet |
| 5 | `snr_calculator.py` | Folded SNR + TLS SDE |
| 6 | `significance.py` | Bootstrap permutation FAP |
| 7 | `false_positive.py` | 5 FP tests (odd/even, secondary, centroid, V-shape, depth-var) |
| 8 | `blend_crosscheck.py` | TIC contamination + Gaia DR3 neighbours |
| 9 | `classifier.py` | Dual-view CNN (2001-pt + 201-pt) + rule ensemble |
| 10 | `fitter.py` | batman MAP → emcee MCMC → posterior uncertainties |

---

## Output files (all in `outputs/`)

| File | Content |
|---|---|
| `TIC_*_diagnostic.png` | 6-panel per-star plot |
| `TIC_*_corner.png` | MCMC posterior corner plot |
| `population_summary.png` | Period-depth scatter + SNR histogram |
| `results_summary.json` | Machine-readable results with all parameters |
| `results_table.csv` | Spreadsheet-friendly summary |
| `report.html` | 3-page ISRO submission report |
| `report.md` | Markdown version |
| `demo_results.csv` | Demo run accuracy table |
| `pipeline.log` | Full execution log |

After training:

| File | Content |
|---|---|
| `models/classifier.h5` | Trained CNN weights |
| `models/calibration_temperature.json` | Temperature scaling + reliability table |
| `outputs/confusion_matrix.png` | Classification accuracy heatmap |
| `outputs/training_history.png` | Loss + accuracy curves |

---

## Known test TIC IDs

| TIC ID | Object | Expected class |
|---|---|---|
| 261136679 | HD 21749b | planet_transit |
| 388857263 | Known EB | eclipsing_binary |
| 100100827 | TOI candidate | planet_transit |

---

## Architecture (what makes this different)

1. **TLS over BLS** — physically accurate limb-darkened template, ~10-15% better sensitivity
2. **Wotan over SG** — robust biweight detrending, no polynomial artifacts
3. **Starspot screen** — explicit Lomb-Scargle variability routing before CNN
4. **Dual-view CNN** — global (2001-pt) + local (201-pt) views, mirrors AstroNet
5. **Bootstrap FAP** — assumption-free permutation significance, not analytic approximation
6. **Blend cross-check** — TIC contamination + Gaia DR3 neighbour, not centroid alone
7. **MCMC posteriors** — full credible intervals, not point estimates
8. **Calibrated confidence** — temperature scaling, reliability table, not raw softmax
9. **Multi-planet search** — iterative mask + re-search
10. **Independent validation** — accuracy from real catalogs not training distribution

---

## References

- Shallue & Vanderburg (2018) — AstroNet dual-view CNN
- Hippke & Heller (2019) — Transit Least Squares
- Hippke et al. (2019) — Wotan detrending
- Kreidberg (2015) — batman transit model
- Foreman-Mackey et al. (2013) — emcee MCMC
- Prša et al. (2022) — TESS EB catalog
- Guo et al. (2017) — neural network confidence calibration

---

**Team JyotirVega | Aurixys | LOGMIEER, Nashik | SPPU**
*PI: Vishal Shivaji Patil*
