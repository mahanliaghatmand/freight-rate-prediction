"""
Freight rate prediction pipeline.

What this script does, step by step:
  1. Loads the labeled training data (data/train_test.csv).
  2. Cleans it (fixes bad weight values, handles missing values).
  3. Builds a small set of model features.
  4. Splits the data by TIME (not randomly) to validate the model the same
     way it will really be used: trained on the past, tested on the future.
  5. Trains a Gradient Boosting model and prints validation metrics.
  6. Retrains the model on ALL available labeled data.
  7. Predicts rates for the 12,000 loads in data/validation.csv and saves
     validation_predictions.csv.
  8. Predicts rates for the 31 fixed December loads in
     data/december_chart_inputs.csv and saves the completed file.

Run it with:
    python src/train.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

FEATURE_COLUMNS = [
    "distance",
    "weight",
    "equipment",
    "month",
    "day_of_week",
]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_frame(df: pd.DataFrame, weight_median: float | None = None) -> tuple[pd.DataFrame, float]:
    """Fix known data-quality issues.

    Issue 1: some `weight` values are negative (a sign-entry error, since a
    physical shipment weight cannot be negative). We fix this by taking the
    absolute value instead of dropping the rows, so we don't throw away
    otherwise-good loads.

    Issue 2: `weight` and `market_index` have missing values. We fill weight
    with the median weight (computed only on the training split, then
    reused everywhere, so no information leaks from validation into
    training). `market_index` is dropped entirely from the model (see
    README for why).
    """
    df = df.copy()

    df["weight"] = df["weight"].abs()

    if weight_median is None:
        weight_median = float(df["weight"].median())
    df["weight"] = df["weight"].fillna(weight_median)

    return df, weight_median


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dates = pd.to_datetime(df["date"])
    df["month"] = dates.dt.month
    df["day_of_week"] = dates.dt.dayofweek
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select and encode the columns the model actually uses.

    Design decision: we deliberately do NOT use the `pickup` / `delivery`
    city names as categorical features. The validation set contains 8
    cities (e.g. Chicago, San Diego, Charlotte...) that never appear in the
    training data at all. A model that memorizes city names would have no
    idea what to do with those loads. `distance`, `pickup_lat/lon` and
    `delivery_lat/lon` already capture "where" a load travels in a way
    that generalizes to new cities, so we rely on `distance` instead.

    Design decision: we also drop `market_index` and `quote_signal`. Both
    have very weak correlation with the target (|r| < 0.04 on the training
    data) and, importantly, `market_index` is missing for some rows while
    the December assessment file doesn't include either column at all. Since
    they add close to no predictive value, excluding them keeps the model
    simple and lets the same model score every file we're given, including
    December, without having to invent values for missing columns.
    """
    encoded = pd.get_dummies(df["equipment"], prefix="equipment")
    features = pd.concat(
        [df[["distance", "weight", "month", "day_of_week"]], encoded],
        axis=1,
    )
    return features


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    print(f"[{label}] MAE=${mae:,.2f}  RMSE=${rmse:,.2f}  R2={r2:.4f}")


def main() -> None:
    train_raw = pd.read_csv(TRAIN_PATH)
    train_raw = add_date_features(train_raw)

    # ------------------------------------------------------------------
    # Time-based validation split.
    #
    # The labeled data covers Jan-Oct 2025. The real validation.csv we
    # must score is Nov-Dec 2025, i.e. the FUTURE relative to training.
    # A random 80/20 split would let the model "peek" at rows from the
    # same week it's tested on, which is unrealistically easy and would
    # overstate how well the model generalizes forward in time. Instead
    # we hold out the last two months of the labeled data (Sep-Oct) as a
    # validation set and train on Jan-Aug, mirroring how the model will
    # actually be used.
    # ------------------------------------------------------------------
    train_raw["date_parsed"] = pd.to_datetime(train_raw["date"])
    cutoff = pd.Timestamp("2025-09-01")
    fit_part = train_raw[train_raw["date_parsed"] < cutoff]
    holdout_part = train_raw[train_raw["date_parsed"] >= cutoff]

    fit_part, weight_median = clean_frame(fit_part)
    holdout_part, _ = clean_frame(holdout_part, weight_median=weight_median)

    X_fit = build_features(fit_part)
    y_fit = fit_part["posted_rate"].values
    X_holdout = build_features(holdout_part).reindex(columns=X_fit.columns, fill_value=0)
    y_holdout = holdout_part["posted_rate"].values

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=0.1,
        random_state=42,
    )
    model.fit(X_fit, y_fit)

    holdout_pred = model.predict(X_holdout)
    print("Validation strategy: trained on Jan-Aug 2025, held out Sep-Oct 2025.")
    evaluate(y_holdout, holdout_pred, "Sep-Oct 2025 holdout")

    # A naive "distance-only average rate" baseline, to show the model is
    # actually adding value beyond a simple rule of thumb.
    baseline_rate_per_mile = (fit_part["posted_rate"] / fit_part["distance"]).mean()
    baseline_pred = holdout_part["distance"].values * baseline_rate_per_mile
    evaluate(y_holdout, baseline_pred, "Baseline (avg rate/mile)")

    # ------------------------------------------------------------------
    # Retrain on ALL labeled data before producing final predictions.
    # ------------------------------------------------------------------
    full_clean, full_weight_median = clean_frame(train_raw)
    X_full = build_features(full_clean)
    y_full = full_clean["posted_rate"].values

    final_model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        l2_regularization=0.1,
        random_state=42,
    )
    final_model.fit(X_full, y_full)

    # ------------------------------------------------------------------
    # Predict on validation.csv (the 12,000 loads Spotter will score).
    # ------------------------------------------------------------------
    validation_raw = pd.read_csv(VALIDATION_PATH)
    validation_raw = add_date_features(validation_raw)
    validation_clean, _ = clean_frame(validation_raw, weight_median=full_weight_median)
    X_validation = build_features(validation_clean).reindex(columns=X_full.columns, fill_value=0)
    validation_predictions = final_model.predict(X_validation)
    validation_predictions = np.clip(validation_predictions, 1.0, None)  # rates must stay positive

    template = pd.read_csv(VALIDATION_TEMPLATE_PATH)
    template["predicted_rate"] = validation_predictions
    validation_out_path = OUT_DIR / "validation_predictions.csv"
    template.to_csv(validation_out_path, index=False)
    print(f"Saved {validation_out_path}")

    # ------------------------------------------------------------------
    # Predict on the fixed December route (31 days, same lane every day).
    # ------------------------------------------------------------------
    december_raw = pd.read_csv(DECEMBER_PATH)
    december_raw = add_date_features(december_raw)
    december_clean, _ = clean_frame(december_raw, weight_median=full_weight_median)
    X_december = build_features(december_clean).reindex(columns=X_full.columns, fill_value=0)
    december_predictions = final_model.predict(X_december)
    december_predictions = np.clip(december_predictions, 1.0, None)

    december_out = december_raw[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
    december_out["predicted_rate"] = december_predictions
    december_out_path = OUT_DIR / "december_chart_inputs.csv"
    december_out.to_csv(december_out_path, index=False)
    print(f"Saved {december_out_path}")


if __name__ == "__main__":
    main()
