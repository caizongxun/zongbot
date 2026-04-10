"""Step 6: LGBM Model Training (Long and Short models).

Strict temporal split - no data leakage.
Long model: learns swing_high(+1) and flat(0) patterns.
Short model: learns swing_low(-1) and flat(0) patterns.

Model naming: v{year}.Q{quarter}.{version}
  - version v2025.Q1.0 -> train 2023-01-01 to 2024-12-31, oos 2025-01-01 to 2025-03-31

Usage:
    python pipeline/step6_train.py \
        --version v2025.Q1.0 \
        --symbol BTCUSDT \
        --features_dir data/features \
        --models_dir models \
        --label_interval 4h \
        --n_trials 50
"""

import argparse
import json
import logging
import re
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

VERSION_PATTERN = re.compile(r"v(\d{4})\.Q([1-4])\.(\d+)")
QUARTER_MONTHS = {1: (1, 3), 2: (4, 6), 3: (7, 9), 4: (10, 12)}


def parse_version(version: str) -> Tuple[int, int, int]:
    m = VERSION_PATTERN.match(version)
    if not m:
        raise ValueError(f"Invalid version format: {version}. Expected v{{year}}.Q{{1-4}}.{{n}}")
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def get_date_ranges(version: str) -> Tuple[str, str, str, str]:
    """Returns (train_start, train_end, oos_start, oos_end) from version string."""
    year, quarter, _ = parse_version(version)
    oos_start_month, oos_end_month = QUARTER_MONTHS[quarter]
    oos_start = date(year, oos_start_month, 1)
    # OOS end = last day of oos_end_month
    if oos_end_month == 12:
        oos_end = date(year, 12, 31)
    else:
        oos_end = date(year, oos_end_month + 1, 1) - timedelta(days=1)
    # Train: 2 years rolling back from oos_start
    train_end = oos_start - timedelta(days=1)
    train_start = date(train_end.year - 2, train_end.month, train_end.day) + timedelta(days=1)
    return (
        train_start.isoformat(), train_end.isoformat(),
        oos_start.isoformat(), oos_end.isoformat()
    )


def load_features(features_dir: Path, symbol: str, label_interval: str) -> pd.DataFrame:
    path = features_dir / symbol / f"{symbol}_{label_interval}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}. Run step5 first.")
    df = pd.read_parquet(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    return df.sort_values("open_time").reset_index(drop=True)


def split_data(df: pd.DataFrame, train_start: str, train_end: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict temporal split - no overlap, no shuffle."""
    ts = pd.Timestamp(train_start, tz="UTC")
    te = pd.Timestamp(train_end, tz="UTC") + pd.Timedelta(days=1)
    train = df[(df["open_time"] >= ts) & (df["open_time"] < te)].copy()
    oos = df[df["open_time"] >= te].copy()
    return train, oos


def prepare_xy(
    df: pd.DataFrame,
    model_type: str,  # "long" or "short"
    meta_cols: list,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X, y for long or short model."""
    if model_type == "long":
        # Long: label 1 (swing_high) = target, 0 (flat) = negative
        # We EXCLUDE label -1 (swing_low) from long model training
        df_filtered = df[df["label"] != -1].copy()
        y = (df_filtered["label"] == 1).astype(int)
    elif model_type == "short":
        # Short: label -1 (swing_low) = target, 0 (flat) = negative
        # We EXCLUDE label +1 (swing_high) from short model training
        df_filtered = df[df["label"] != 1].copy()
        y = (df_filtered["label"] == -1).astype(int)
    else:
        raise ValueError(f"model_type must be 'long' or 'short', got: {model_type}")

    feature_cols = [c for c in df_filtered.columns if c not in meta_cols]
    X = df_filtered[feature_cols].copy()
    # Fill NaN with median (computed on train, applied consistently)
    X = X.fillna(X.median())
    return X, y, df_filtered


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str,
    n_trials: int = 50,
    use_optuna: bool = True,
) -> object:
    """Train LGBM with optional Optuna hyperparameter optimization."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise ImportError("lightgbm not installed. Run: pip install lightgbm")

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos = n_neg / (n_pos + 1)
    log.info(f"[Step6] {model_type} | train size={len(y_train)} | pos={n_pos} neg={n_neg} scale_pos={scale_pos:.2f}")

    default_params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=64,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=30,
        lambda_l1=0.1,
        lambda_l2=0.1,
        scale_pos_weight=scale_pos,
        n_estimators=1000,
        early_stopping_rounds=50,
        verbose=-1,
        random_state=42,
    )

    if use_optuna and n_trials > 0:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)

            def objective(trial):
                params = dict(
                    objective="binary",
                    metric="auc",
                    boosting_type="gbdt",
                    num_leaves=trial.suggest_int("num_leaves", 16, 128),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                    feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
                    bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                    bagging_freq=5,
                    min_child_samples=trial.suggest_int("min_child_samples", 10, 60),
                    lambda_l1=trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
                    lambda_l2=trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
                    scale_pos_weight=scale_pos,
                    n_estimators=500,
                    verbose=-1,
                    random_state=42,
                )
                # Walk-forward CV (last 20% of train as val)
                split = int(len(X_train) * 0.8)
                X_cv, X_val = X_train.iloc[:split], X_train.iloc[split:]
                y_cv, y_val = y_train.iloc[:split], y_train.iloc[split:]
                model = lgb.LGBMClassifier(**params)
                model.fit(X_cv, y_cv,
                          eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
                from sklearn.metrics import roc_auc_score
                pred = model.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, pred)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
            best = study.best_params
            log.info(f"[Step6] Optuna best AUC={study.best_value:.4f} | params={best}")
            default_params.update(best)
        except ImportError:
            log.warning("Optuna not installed, using default params.")

    # Final training on full train set
    split = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:split], X_train.iloc[split:]
    y_tr, y_val = y_train.iloc[:split], y_train.iloc[split:]

    model = lgb.LGBMClassifier(**{k: v for k, v in default_params.items()
                                  if k not in ["early_stopping_rounds"]})
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(default_params.get("early_stopping_rounds", 50), verbose=False),
                   lgb.log_evaluation(50)],
    )
    return model


def run(
    version: str,
    symbol: str,
    features_dir: str,
    models_dir: str,
    label_interval: str = "4h",
    n_trials: int = 50,
    skip: bool = False,
):
    if skip:
        log.info("[Step6] skip=True, skipping training.")
        return

    train_start, train_end, oos_start, oos_end = get_date_ranges(version)
    log.info(f"[Step6] Version: {version}")
    log.info(f"[Step6] Train: {train_start} -> {train_end}")
    log.info(f"[Step6] OOS:   {oos_start} -> {oos_end}")

    feat_path = Path(features_dir)
    model_path = Path(models_dir) / version
    model_path.mkdir(parents=True, exist_ok=True)

    df = load_features(feat_path, symbol, label_interval)
    log.info(f"[Step6] Loaded {len(df)} rows of features")

    train_df, oos_df = split_data(df, train_start, train_end)
    log.info(f"[Step6] Train rows: {len(train_df)} | OOS rows: {len(oos_df)}")

    if len(train_df) == 0:
        log.error("No training data found in specified date range.")
        sys.exit(1)

    META_COLS = ["open_time", "label", "close", "high", "low"]

    # Save OOS data for backtesting
    oos_path = model_path / f"{symbol}_oos.parquet"
    oos_df.to_parquet(oos_path, index=False)
    log.info(f"[Step6] OOS data saved: {oos_path}")

    for model_type in ["long", "short"]:
        log.info(f"\n[Step6] Training {model_type.upper()} model...")
        X_train, y_train, filtered_train = prepare_xy(train_df, model_type, META_COLS)
        model = train_lgbm(X_train, y_train, model_type, n_trials=n_trials)

        # Save model
        try:
            import joblib
            mfile = model_path / f"{symbol}_{model_type}_model.pkl"
            joblib.dump(model, mfile)
            log.info(f"[Step6] Saved {model_type} model -> {mfile}")
        except Exception as e:
            log.error(f"Failed to save model: {e}")

        # Save feature importance
        try:
            fi = pd.DataFrame({
                "feature": X_train.columns,
                "importance": model.feature_importances_,
            }).sort_values("importance", ascending=False)
            fi.to_csv(model_path / f"{symbol}_{model_type}_feature_importance.csv", index=False)
            log.info(f"[Step6] Top 10 features ({model_type}): {fi['feature'].head(10).tolist()}")
        except Exception:
            pass

    # Save metadata
    meta = {
        "version": version,
        "symbol": symbol,
        "label_interval": label_interval,
        "train_start": train_start,
        "train_end": train_end,
        "oos_start": oos_start,
        "oos_end": oos_end,
        "train_rows": len(train_df),
        "oos_rows": len(oos_df),
        "feature_count": len([c for c in df.columns if c not in META_COLS]),
    }
    with open(model_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"[Step6] Training complete. Models saved in {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step6: LGBM Model Training")
    parser.add_argument("--version", required=True, help="e.g. v2025.Q1.0")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--label_interval", default="4h")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    run(
        version=args.version,
        symbol=args.symbol,
        features_dir=args.features_dir,
        models_dir=args.models_dir,
        label_interval=args.label_interval,
        n_trials=args.n_trials,
        skip=args.skip,
    )
