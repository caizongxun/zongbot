"""Step 3: PySR Symbolic Regression for market background discovery.

Discovers symbolic formulas across multiple market perspectives:
  - trade_context: volume/trade background
  - momentum: price momentum proxies
  - sentiment: fear/greed sentiment proxies
  - microstructure: order flow imbalance
  - volatility: realized vol regime

Usage:
    python pipeline/step3_pysr.py \
        --symbol BTCUSDT \
        --processed_dir data/processed \
        --pysr_output_dir data/pysr \
        --base_interval 1h \
        --perspectives trade_context momentum sentiment microstructure volatility \
        --populations 20 \
        --niterations 100 \
        --max_depth 7 \
        --skip False
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_klines(processed_dir: Path, symbol: str, interval: str) -> pd.DataFrame:
    p = processed_dir / "klines" / symbol / f"{symbol}_{interval}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"klines not found: {p}")
    df = pd.read_parquet(p)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute base market features for PySR input."""
    c = df["close"].values
    h = df["high"].values
    lo = df["low"].values
    v = df["volume"].values
    trd = df["trades"].fillna(0).values.astype(float)
    tbv = df["taker_buy_volume"].values
    qv = df["quote_volume"].values

    eps = 1e-10
    feat = pd.DataFrame(index=df.index)

    # Returns
    feat["ret1"] = pd.Series(c).pct_change(1)
    feat["ret3"] = pd.Series(c).pct_change(3)
    feat["ret5"] = pd.Series(c).pct_change(5)
    feat["ret10"] = pd.Series(c).pct_change(10)
    feat["ret20"] = pd.Series(c).pct_change(20)

    # Volume ratios
    vol_ma = pd.Series(v).rolling(20).mean()
    feat["vol_ratio"] = pd.Series(v) / (vol_ma + eps)
    feat["taker_buy_ratio"] = pd.Series(tbv) / (pd.Series(v) + eps)

    # Order flow imbalance (delta)
    feat["delta"] = pd.Series(tbv) - (pd.Series(v) - pd.Series(tbv))
    feat["delta_ma5"] = feat["delta"].rolling(5).mean()
    feat["cvd"] = feat["delta"].cumsum()  # Cumulative Volume Delta

    # Volatility
    log_ret = np.log(pd.Series(c) / pd.Series(c).shift(1))
    feat["realized_vol_10"] = log_ret.rolling(10).std() * np.sqrt(252 * 24)
    feat["realized_vol_20"] = log_ret.rolling(20).std() * np.sqrt(252 * 24)
    feat["atr_14"] = pd.Series(
        np.maximum(h - lo,
                   np.maximum(np.abs(h - np.roll(c, 1)), np.abs(lo - np.roll(c, 1))))
    ).rolling(14).mean()

    # Price structure
    feat["hl_range"] = (pd.Series(h) - pd.Series(lo)) / (pd.Series(c) + eps)
    feat["body_ratio"] = np.abs(pd.Series(c) - pd.Series(df["open"])) / (pd.Series(h) - pd.Series(lo) + eps)

    # Trade intensity
    trd_ma = pd.Series(trd).rolling(20).mean()
    feat["trade_intensity"] = pd.Series(trd) / (trd_ma + eps)
    feat["avg_trade_size"] = pd.Series(qv) / (pd.Series(trd) + eps)

    # Momentum indicators
    feat["rsi_14"] = _rsi(pd.Series(c), 14)
    feat["rsi_28"] = _rsi(pd.Series(c), 28)
    feat["macd_hist"] = _macd_hist(pd.Series(c))
    feat["cci_20"] = _cci(pd.Series(h), pd.Series(lo), pd.Series(c), 20)

    # Funding rate if available (will be merged at feature step, placeholder zeros here)
    feat["funding_proxy"] = 0.0

    return feat


def _rsi(s: pd.Series, period: int) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _macd_hist(s: pd.Series) -> pd.Series:
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd - signal


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    tp = (high + low + close) / 3
    ma = tp.rolling(period).mean()
    mad = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - ma) / (0.015 * mad + 1e-10)


PERSPECTIVE_TARGETS = {
    "trade_context": {
        "desc": "Predict future volume ratio shift (market activity regime)",
        "target_fn": lambda f: f["vol_ratio"].shift(-5).rolling(5).mean(),
        "feature_cols": ["vol_ratio", "taker_buy_ratio", "delta", "delta_ma5", "trade_intensity", "avg_trade_size", "atr_14"],
    },
    "momentum": {
        "desc": "Predict signed 10-bar forward return",
        "target_fn": lambda f: f["ret10"].shift(-10),
        "feature_cols": ["ret1", "ret3", "ret5", "rsi_14", "rsi_28", "macd_hist", "cci_20", "vol_ratio"],
    },
    "sentiment": {
        "desc": "Predict taker buy ratio shift (market fear/greed proxy)",
        "target_fn": lambda f: f["taker_buy_ratio"].shift(-10).rolling(10).mean(),
        "feature_cols": ["taker_buy_ratio", "delta_ma5", "rsi_14", "vol_ratio", "realized_vol_10", "funding_proxy"],
    },
    "microstructure": {
        "desc": "Predict CVD acceleration (order flow regime)",
        "target_fn": lambda f: f["delta"].shift(-5).rolling(5).mean() - f["delta"].rolling(5).mean(),
        "feature_cols": ["delta", "delta_ma5", "cvd", "taker_buy_ratio", "trade_intensity", "hl_range", "body_ratio"],
    },
    "volatility": {
        "desc": "Predict realized vol expansion/contraction",
        "target_fn": lambda f: (f["realized_vol_10"].shift(-10) - f["realized_vol_10"]) / (f["realized_vol_10"] + 1e-10),
        "feature_cols": ["realized_vol_10", "realized_vol_20", "atr_14", "hl_range", "ret5", "vol_ratio"],
    },
}


def run_pysr_perspective(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    out_dir: Path,
    populations: int,
    niterations: int,
    max_depth: int,
):
    """Run PySR for a single perspective and save results."""
    try:
        from pysr import PySRRegressor
    except ImportError:
        log.error("PySR not installed. Run: pip install pysr")
        return None

    log.info(f"[Step3] Running PySR for perspective: {name} | X={X.shape}")

    model = PySRRegressor(
        niterations=niterations,
        populations=populations,
        maxsize=max_depth * 3,
        model_selection="best",
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "cube", "sqrt", "abs", "log", "neg"],
        verbosity=0,
        progress=True,
        random_state=42,
        deterministic=True,
        parallelism="multithreading",
        output_jax_format=False,
        extra_sympy_mappings={"cube": lambda x: x**3},
    )

    model.fit(X, y, variable_names=feature_names)

    equations = model.equations_
    out_path = out_dir / f"pysr_{name}.csv"
    equations.to_csv(out_path, index=False)
    log.info(f"[Step3] Saved PySR equations for {name} -> {out_path}")

    # Save best expression metadata
    meta = {
        "perspective": name,
        "best_equation": str(model.sympy()),
        "best_loss": float(equations["loss"].min()),
        "feature_names": feature_names,
    }
    meta_path = out_dir / f"pysr_{name}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"[Step3] Best equation for {name}: {meta['best_equation']}")
    return model


def run(
    symbol: str,
    processed_dir: str,
    pysr_output_dir: str,
    base_interval: str = "1h",
    perspectives: List[str] = None,
    populations: int = 20,
    niterations: int = 100,
    max_depth: int = 7,
    max_samples: int = 20000,
    skip: bool = False,
):
    if skip:
        log.info("[Step3] skip=True, skipping PySR evolution.")
        return

    if perspectives is None:
        perspectives = list(PERSPECTIVE_TARGETS.keys())

    proc_path = Path(processed_dir)
    out_dir = Path(pysr_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_klines(proc_path, symbol, base_interval)
    features = compute_base_features(df)

    for persp in perspectives:
        if persp not in PERSPECTIVE_TARGETS:
            log.warning(f"Unknown perspective: {persp}, skipping.")
            continue

        cfg = PERSPECTIVE_TARGETS[persp]
        target = cfg["target_fn"](features)
        feat_cols = cfg["feature_cols"]

        combined = features[feat_cols].copy()
        combined["__target__"] = target
        combined = combined.dropna()

        if len(combined) > max_samples:
            combined = combined.sample(max_samples, random_state=42)

        X = combined[feat_cols].values.astype(float)
        y = combined["__target__"].values.astype(float)

        run_pysr_perspective(
            name=persp,
            X=X,
            y=y,
            feature_names=feat_cols,
            out_dir=out_dir,
            populations=populations,
            niterations=niterations,
            max_depth=max_depth,
        )

    log.info("[Step3] PySR evolution complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step3: PySR Symbolic Regression")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--pysr_output_dir", default="data/pysr")
    parser.add_argument("--base_interval", default="1h")
    parser.add_argument("--perspectives", nargs="+", default=list(PERSPECTIVE_TARGETS.keys()))
    parser.add_argument("--populations", type=int, default=20)
    parser.add_argument("--niterations", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=7)
    parser.add_argument("--max_samples", type=int, default=20000)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    run(
        symbol=args.symbol,
        processed_dir=args.processed_dir,
        pysr_output_dir=args.pysr_output_dir,
        base_interval=args.base_interval,
        perspectives=args.perspectives,
        populations=args.populations,
        niterations=args.niterations,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        skip=args.skip,
    )
