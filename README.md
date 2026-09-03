# Freight Rate Prediction

A machine learning pipeline that predicts truckload freight rates from shipment attributes (distance, weight, equipment type, and pickup timing), built as part of a machine learning engineering assessment.

## Overview

Given historical freight loads with a known `posted_rate`, the goal is to train a model that can predict rates for **new, unseen loads** — including loads on lanes/cities the model has never seen and dates months after the training window ends. The project covers the full pipeline: data cleaning, feature engineering, time-aware validation, model training, and scoring on two held-out prediction tasks.

## Problem Setup

- **Training data**: `data/train_test.csv` — 48,000 labeled loads from January–October 2025, including the target `posted_rate`.
- **Task 1 — Validation set**: `data/validation.csv` — 12,000 unlabeled loads from November–December 2025 to be scored and submitted as `load_id, predicted_rate`.
- **Task 2 — December chart**: `data/december_chart_inputs.csv` — a single fixed lane (Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb) repeated once per day for all 31 days of December 2025, used to visualize how predicted rate moves over the month with every input held constant except the date.

Both prediction targets fall **after** the end of the labeled training period, which shapes every modeling decision below.

## Data Cleaning

Two data-quality issues were identified and handled explicitly in `src/train.py`:

| Issue | Scope | Fix |
|---|---|---|
| Negative `weight` values | ~0.6% of rows | Sign-entry error — corrected with `abs()` rather than dropping rows |
| Missing `weight` / `market_index` | ~0.6% / ~0.8% of rows | `weight` imputed with the **training-split median** (never validation data, to avoid leakage); `market_index` dropped entirely (see Feature Selection) |

## Feature Engineering

**Features used**: `distance`, `weight`, `equipment` (one-hot encoded), `month`, `day_of_week` (both derived from `date`).

**Deliberately excluded:**

- **`pickup` / `delivery` city names** — the validation set contains 8 cities that never appear in training at all (Chicago, San Diego, Charlotte, Knoxville, Jackson, Norfolk, Laredo, Allentown). A model that learns per-city effects would have no signal for these loads. `distance` already encodes "how far the load travels" in a way that generalizes to unseen cities, so city identity is dropped in favor of it.
- **`market_index` and `quote_signal`** — both correlate very weakly with `posted_rate` (`|r| < 0.04`) in the training data, and neither column exists in `december_chart_inputs.csv`. Dropping them lets one single trained model score every provided file without having to invent values for missing columns.

## Validation Strategy

The labeled data ends in October 2025; both prediction targets start in November. A random 80/20 split of the labeled data would let the model validate on rows from the same weeks it trained on — an easier problem than predicting genuinely unseen future months, and one that would overstate real-world accuracy.

Instead, a **time-based split** is used:

- **Fit**: January–August 2025
- **Holdout**: September–October 2025 (never seen during training)

This mirrors how the model is actually used — trained on the past, scored on the future — and gives an honest estimate of forward-looking performance. Once validated, the final model is **retrained on all available labeled data (Jan–Oct)** before producing the submitted predictions.

## Model

**Gradient Boosting** (`sklearn.ensemble.HistGradientBoostingRegressor`), chosen over a plain linear model because equipment type does not affect rate/mile additively — Reefer and Flatbed carry different premiums that interact with distance and season — and boosted trees capture that without manual feature crossing.

```python
HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.08,
    max_iter=300,
    l2_regularization=0.1,
    random_state=42,
)
```

## Results

Evaluated on the Sep–Oct 2025 holdout, against a naive "average rate per mile × distance" baseline:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (avg rate/mile) | $290.60 | $709.45 | 0.784 |
| Gradient Boosting (this model) | $131.57 | $633.57 | 0.828 |

The model cuts average error by roughly **55%** versus the baseline, confirming that weight, equipment type, and seasonality carry real predictive signal beyond distance alone.

## A Note on the December Chart

The December predicted-rate curve comes out **nearly flat**. This is expected, not a bug: the training data only spans pickup months January–October, so month 12 (December) falls outside the range the model was ever trained on. Tree-based models don't extrapolate trends beyond their training range — they fall back on the closest pattern they learned. The small remaining week-to-week wiggle comes entirely from the `day_of_week` feature. The flatness is itself an honest signal that the model recognizes December as out-of-distribution rather than guessing wildly.

## Project Structure

```
.
├── data/
│   ├── train_test.csv                      # labeled training data (Jan–Oct 2025)
│   ├── validation.csv                      # 12,000 loads to score (Nov–Dec 2025)
│   ├── validation_predictions_template.csv # output template (load_id, predicted_rate)
│   └── december_chart_inputs.csv           # fixed lane, 31 daily rows for Dec 2025
├── src/
│   └── train.py          # cleaning, feature engineering, training, validation, prediction
├── score.py               # validates submission files and renders the December chart
├── make_report.py         # generates a PDF write-up of the approach and results
├── outputs/                # generated predictions, chart, and report land here
└── requirements.txt
```

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model, print validation metrics, and generate both prediction files:

```bash
python src/train.py
```

This writes `outputs/validation_predictions.csv` (12,000 scored loads) and `outputs/december_chart_inputs.csv` (the completed 31-row December file).

Validate the outputs and render the December chart:

```bash
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv \
  --output-dir outputs/scorer_results
```

`score.py` checks row counts, ID coverage, column order, and the fixed values required for the December file, then saves `candidate_december.png`.

Optionally, generate a PDF summary report (requires `reportlab`, not listed in `requirements.txt`):

```bash
pip install reportlab
python make_report.py
```

## Requirements

```
matplotlib>=3.8,<4
numpy>=1.26,<3
pandas>=2.0,<3
scikit-learn>=1.3,<2
```
