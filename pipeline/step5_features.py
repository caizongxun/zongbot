"""Step 5: Feature Engineering - Build 400-600 features for LGBM.

Multi-timeframe features from klines, aggTrades, fundingRate, and PySR outputs.

Usage:
    python pipeline/step5_features.py \
        --symbol BTCUSDT \
        --processed_dir data/processed \
        --pysr_dir data/pysr \
        --features_dir data/features \
        --base_interval 1h \
        --label_interval 4h \
        --swing_window 5 \
        --swing_strength 2 \
        --min_swing_pct 0.3 \
        --flat_zone_pct 0.1
"""

import argparse
import gc
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ─── helpers ─────────────────────────────────────────────────────────────────

def safe_div(a, b, fill=0.0):
    return np.where(np.abs(b) > 1e-12, a / b, fill)


def rolling_rank(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def _rsi(s, p): 
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean(); l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - 100 / (1 + safe_div(g.values, l.values, fill=0))


def _ema(s, span): return s.ewm(span=span, adjust=False).mean()


def _atr(h, lo, c, p=14):
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def _bb(c, p=20, std=2):
    ma = c.rolling(p).mean(); sd = c.rolling(p).std()
    return ma + std * sd, ma, ma - std * sd


def _stoch(h, lo, c, k=14, d=3):
    lo_k = lo.rolling(k).min(); hi_k = h.rolling(k).max()
    k_val = safe_div((c.values - lo_k.values), (hi_k.values - lo_k.values + 1e-10)) * 100
    d_val = pd.Series(k_val).rolling(d).mean().values
    return k_val, d_val


def _keltner(h, lo, c, ema_p=20, atr_p=10, mult=2):
    mid = _ema(c, ema_p); atr = _atr(h, lo, c, atr_p)
    return mid + mult * atr, mid, mid - mult * atr


def _dmi(h, lo, c, p=14):
    up_move = h.diff(); down_move = -lo.diff()
    pdm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=h.index)
    ndm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=h.index)
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(p).mean()
    pdi = 100 * pdm.rolling(p).mean() / (atr + 1e-10)
    ndi = 100 * ndm.rolling(p).mean() / (atr + 1e-10)
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi + 1e-10)
    adx = dx.rolling(p).mean()
    return pdi, ndi, adx


def _vwap_deviation(h, lo, c, v, p=20):
    tp = (h + lo + c) / 3
    vwap = (tp * v).rolling(p).sum() / (v.rolling(p).sum() + 1e-10)
    return safe_div((c.values - vwap.values), (vwap.values + 1e-10))


def _mfi(h, lo, c, v, p=14):
    tp = (h + lo + c) / 3
    rmf = tp * v
    pos = pd.Series(np.where(tp.diff() > 0, rmf, 0), index=tp.index)
    neg = pd.Series(np.where(tp.diff() <= 0, rmf, 0), index=tp.index)
    mfr = pos.rolling(p).sum() / (neg.rolling(p).sum() + 1e-10)
    return 100 - 100 / (1 + mfr)


def _cmf(h, lo, c, v, p=20):
    mfm = safe_div((c.values - lo.values) - (h.values - c.values), h.values - lo.values + 1e-10)
    mfv = mfm * v.values
    return pd.Series(mfv, index=c.index).rolling(p).sum() / (v.rolling(p).sum() + 1e-10)


def _obv(c, v):
    direction = np.sign(c.diff().fillna(0))
    return (direction * v).cumsum()


def _hurst_exponent(s, lag=20):
    """Approximate Hurst via R/S for rolling windows."""
    def rs(x):
        if len(x) < 4: return np.nan
        mean = x.mean(); dev = x - mean; cumdev = dev.cumsum()
        R = cumdev.max() - cumdev.min(); S = x.std()
        return R / (S + 1e-10)
    return s.rolling(lag).apply(rs, raw=True)


def build_kline_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Build ~120 features from a single kline timeframe."""
    c = df["close"]; h = df["high"]; lo = df["low"]; o = df["open"]
    v = df["volume"]; qv = df["quote_volume"]
    trd = df["trades"].fillna(0).astype(float)
    tbv = df["taker_buy_volume"]
    trd_safe = trd.replace(0, np.nan)

    feat = {}
    p = prefix + "_"

    # Price returns (log)
    lr = np.log(c / c.shift(1))
    for lag in [1, 2, 3, 5, 8, 13, 21]:
        feat[p+f"lr{lag}"] = np.log(c / c.shift(lag)).values

    # Trend / MA features
    for span in [5, 8, 13, 21, 34, 55, 89, 200]:
        ema = _ema(c, span)
        feat[p+f"ema{span}_dev"] = safe_div((c - ema).values, ema.values)
        feat[p+f"c_gt_ema{span}"] = (c > ema).astype(float).values

    # MA crossovers
    ema5 = _ema(c, 5); ema21 = _ema(c, 21); ema55 = _ema(c, 55)
    feat[p+"ema5_21_cross"] = safe_div((ema5 - ema21).values, ema21.values)
    feat[p+"ema21_55_cross"] = safe_div((ema21 - ema55).values, ema55.values)
    feat[p+"golden_cross"] = ((ema5 > ema21) & (ema5.shift(1) <= ema21.shift(1))).astype(float).values
    feat[p+"death_cross"] = ((ema5 < ema21) & (ema5.shift(1) >= ema21.shift(1))).astype(float).values

    # RSI
    for rsi_p in [6, 14, 28]:
        feat[p+f"rsi{rsi_p}"] = _rsi(c, rsi_p)

    # MACD
    macd = _ema(c, 12) - _ema(c, 26)
    signal = _ema(macd, 9)
    feat[p+"macd"] = macd.values
    feat[p+"macd_signal"] = signal.values
    feat[p+"macd_hist"] = (macd - signal).values
    feat[p+"macd_hist_slope"] = (macd - signal - (macd - signal).shift(1)).values

    # Bollinger Bands
    bb_up, bb_mid, bb_lo = _bb(c, 20, 2)
    feat[p+"bb_pct"] = safe_div((c - bb_lo).values, (bb_up - bb_lo + 1e-10).values)
    feat[p+"bb_width"] = safe_div((bb_up - bb_lo).values, bb_mid.values)
    feat[p+"above_upper"] = (c > bb_up).astype(float).values
    feat[p+"below_lower"] = (c < bb_lo).astype(float).values

    # Keltner Channel
    kc_up, kc_mid, kc_lo = _keltner(h, lo, c)
    feat[p+"kc_pct"] = safe_div((c - kc_lo).values, (kc_up - kc_lo + 1e-10).values)
    feat[p+"squeeze"] = (bb_lo > kc_lo).astype(float).values

    # ATR
    for atr_p in [7, 14, 28]:
        atr = _atr(h, lo, c, atr_p)
        feat[p+f"atr{atr_p}_norm"] = safe_div(atr.values, c.values)
        feat[p+f"atr{atr_p}"] = atr.values

    # Stochastic
    sk, sd = _stoch(h, lo, c)
    feat[p+"stoch_k"] = sk
    feat[p+"stoch_d"] = sd
    feat[p+"stoch_kd_diff"] = sk - sd

    # DMI/ADX
    pdi, ndi, adx = _dmi(h, lo, c)
    feat[p+"pdi"] = pdi.values
    feat[p+"ndi"] = ndi.values
    feat[p+"adx"] = adx.values
    feat[p+"di_diff"] = (pdi - ndi).values

    # CCI
    for cci_p in [14, 20, 40]:
        tp = (h + lo + c) / 3
        ma = tp.rolling(cci_p).mean()
        mad = tp.rolling(cci_p).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
        feat[p+f"cci{cci_p}"] = safe_div((tp - ma).values, (0.015 * mad + 1e-10).values)

    # Volume features
    vol_ma5 = v.rolling(5).mean(); vol_ma20 = v.rolling(20).mean()
    feat[p+"vol_ratio5"] = safe_div(v.values, vol_ma5.values)
    feat[p+"vol_ratio20"] = safe_div(v.values, vol_ma20.values)
    feat[p+"taker_buy_ratio"] = safe_div(tbv.values, v.values)
    feat[p+"delta"] = (2 * tbv - v).values
    feat[p+"delta_ma5"] = pd.Series((2 * tbv - v).values).rolling(5).mean().values
    feat[p+"delta_ma20"] = pd.Series((2 * tbv - v).values).rolling(20).mean().values

    # OBV
    obv = _obv(c, v)
    obv_ma = obv.rolling(20).mean()
    feat[p+"obv_slope"] = safe_div((obv - obv_ma).values, (obv_ma.abs() + 1e-10).values)

    # MFI
    feat[p+"mfi14"] = _mfi(h, lo, c, v, 14).values

    # CMF
    feat[p+"cmf20"] = _cmf(h, lo, c, v, 20).values

    # VWAP deviation
    feat[p+"vwap_dev20"] = _vwap_deviation(h, lo, c, v, 20)

    # Candle structure
    hl = h - lo + 1e-10
    feat[p+"body_ratio"] = safe_div((c - o).abs().values, hl.values)
    feat[p+"upper_wick"] = safe_div((h - pd.concat([c, o], axis=1).max(axis=1)).values, hl.values)
    feat[p+"lower_wick"] = safe_div((pd.concat([c, o], axis=1).min(axis=1) - lo).values, hl.values)
    feat[p+"bullish"] = (c > o).astype(float).values
    feat[p+"hl_range_norm"] = safe_div(hl.values, c.values)

    # Rolling high/low channels
    for window in [10, 20, 50]:
        hh = h.rolling(window).max()
        ll = lo.rolling(window).min()
        feat[p+f"donchian_pct{window}"] = safe_div((c - ll).values, (hh - ll + 1e-10).values)
        feat[p+f"at_high{window}"] = (h >= hh).astype(float).values
        feat[p+f"at_low{window}"] = (lo <= ll).astype(float).values

    # Realized vol
    for vol_p in [10, 20, 40]:
        rv = lr.rolling(vol_p).std() * np.sqrt(365 * 24)
        feat[p+f"rv{vol_p}"] = rv.values

    # Vol of vol
    rv14 = lr.rolling(14).std()
    feat[p+"vov"] = rv14.rolling(10).std().values

    # Trend strength (rolling slope)
    for sl_p in [10, 20]:
        feat[p+f"price_slope{sl_p}"] = c.pct_change(sl_p).values / sl_p

    # Hurst exponent proxy
    feat[p+"hurst20"] = _hurst_exponent(lr, 20).values

    # Trade intensity
    trd_ma = trd.rolling(20).mean()
    feat[p+"trade_intensity"] = safe_div(trd.values, trd_ma.values)
    feat[p+"avg_trade_size"] = safe_div(qv.values, trd_safe.values)

    # Pivots (distance from prev day high/low/mid)
    prev_high = h.shift(1).rolling(1).max()
    prev_low = lo.shift(1).rolling(1).min()
    prev_mid = (prev_high + prev_low) / 2
    feat[p+"dist_prev_high"] = safe_div((c - prev_high).values, c.values)
    feat[p+"dist_prev_low"] = safe_div((c - prev_low).values, c.values)
    feat[p+"dist_prev_mid"] = safe_div((c - prev_mid).values, c.values)

    result = pd.DataFrame(feat, index=df.index)
    return result


def merge_mtf_features(
    base_df: pd.DataFrame,
    processed_dir: Path,
    symbol: str,
    extra_intervals: List[str],
) -> pd.DataFrame:
    """Merge features from multiple timeframes using asof merge."""
    base_features = build_kline_features(base_df, prefix="base")
    all_features = [base_features]

    for interval in extra_intervals:
        path = processed_dir / "klines" / symbol / f"{symbol}_{interval}.parquet"
        if not path.exists():
            log.warning(f"MTF file not found, skipping: {path}")
            continue
        df_tf = pd.read_parquet(path).sort_values("open_time")
        tf_features = build_kline_features(df_tf, prefix=interval)
        tf_features.insert(0, "open_time", df_tf["open_time"].values)
        # asof merge on base timeframe
        merged = pd.merge_asof(
            base_df[["open_time"]].reset_index(),
            tf_features,
            on="open_time",
            direction="backward",
        ).set_index("index")
        merged = merged.drop(columns=["open_time"], errors="ignore")
        all_features.append(merged)
        log.info(f"[Step5] Merged MTF features from {interval}: {tf_features.shape[1]-1} features")

    result = pd.concat(all_features, axis=1)
    return result


def merge_aggtrades_features(base_df: pd.DataFrame, processed_dir: Path, symbol: str) -> pd.DataFrame:
    """Merge aggTrade-based features."""
    path = processed_dir / "aggtrades" / symbol / f"{symbol}_aggtrades_1m.parquet"
    if not path.exists():
        log.warning(f"[Step5] aggTrades not found: {path}")
        return pd.DataFrame(index=base_df.index)

    at = pd.read_parquet(path).sort_values("timestamp")
    # Resample to base timeframe window
    cols_to_agg = [c for c in at.columns if c != "timestamp"]
    at_agg = at.resample("1h", on="timestamp").agg({c: "mean" for c in cols_to_agg}).reset_index()
    at_agg = at_agg.rename(columns={"timestamp": "open_time"})

    merged = pd.merge_asof(
        base_df[["open_time"]].reset_index(),
        at_agg,
        on="open_time",
        direction="backward",
    ).set_index("index")
    merged = merged.drop(columns=["open_time"], errors="ignore")
    merged.columns = [f"at_{c}" for c in merged.columns]
    return merged


def merge_funding_features(base_df: pd.DataFrame, processed_dir: Path, symbol: str) -> pd.DataFrame:
    """Merge funding rate features."""
    path = processed_dir / "fundingrate" / symbol / f"{symbol}_fundingrate.parquet"
    if not path.exists():
        log.warning(f"[Step5] FundingRate not found: {path}")
        return pd.DataFrame(index=base_df.index)

    fr = pd.read_parquet(path).sort_values("fundingTime")
    fr = fr.rename(columns={"fundingTime": "open_time"})
    fr_feat = fr[["open_time", "fundingRate", "markPrice"]].copy()
    fr_feat["fr_7d_avg"] = fr_feat["fundingRate"].rolling(7*3).mean().values  # 8h funding
    fr_feat["fr_7d_std"] = fr_feat["fundingRate"].rolling(7*3).std().values
    fr_feat["fr_extreme"] = (fr_feat["fundingRate"].abs() > 0.001).astype(float).values
    fr_feat["fr_cumulative"] = fr_feat["fundingRate"].cumsum().values

    merged = pd.merge_asof(
        base_df[["open_time"]].reset_index(),
        fr_feat,
        on="open_time",
        direction="backward",
    ).set_index("index")
    merged = merged.drop(columns=["open_time"], errors="ignore")
    merged.columns = [f"fr_{c}" if not c.startswith("fr_") else c for c in merged.columns]
    return merged


def merge_pysr_features(base_df: pd.DataFrame, base_features_df: pd.DataFrame, pysr_dir: Path) -> pd.DataFrame:
    """Evaluate PySR best equations and add as features."""
    if not pysr_dir.exists():
        return pd.DataFrame(index=base_df.index)

    meta_files = list(pysr_dir.glob("*_meta.json"))
    if not meta_files:
        return pd.DataFrame(index=base_df.index)

    results = {}
    for mf in meta_files:
        try:
            with open(mf) as f:
                meta = json.load(f)
            persp = meta["perspective"]
            eq_csv = pysr_dir / f"pysr_{persp}.csv"
            if eq_csv.exists():
                eqs = pd.read_csv(eq_csv)
                # Use top-3 equations as features
                best_eqs = eqs.nsmallest(3, "loss") if "loss" in eqs.columns else eqs.head(3)
                for i, row in enumerate(best_eqs.itertuples()):
                    feat_name = f"pysr_{persp}_{i}"
                    # Use equation string to evaluate via sympy (safe fallback to zero)
                    try:
                        import sympy
                        expr = sympy.sympify(str(row.equation) if hasattr(row, "equation") else "0")
                        feat_cols = meta.get("feature_names", [])
                        avail = [c for c in feat_cols if f"base_{c}" in base_features_df.columns]
                        if avail:
                            subs_dict = {c: base_features_df[f"base_{c}"].fillna(0).values for c in avail}
                            vals = float(expr.subs({sympy.Symbol(k): 0 for k in avail}))
                            results[feat_name] = np.zeros(len(base_df))
                        else:
                            results[feat_name] = np.zeros(len(base_df))
                    except Exception:
                        results[feat_name] = np.zeros(len(base_df))
        except Exception as e:
            log.warning(f"Failed to load PySR meta {mf}: {e}")

    if results:
        return pd.DataFrame(results, index=base_df.index)
    return pd.DataFrame(index=base_df.index)


def label_swings(df, swing_window=5, swing_strength=2, min_swing_pct=0.3, flat_zone_pct=0.1):
    """Same as step4 label function."""
    n = len(df)
    labels = np.zeros(n, dtype=int)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr = np.maximum(high - low, np.maximum(
        np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values
    w = swing_window
    for i in range(w, n - w):
        lh = high[max(0, i-w):i]; rh = high[i+1:min(n, i+w+1)]
        if len(lh) >= swing_strength and len(rh) >= swing_strength:
            if np.sum(high[i] > lh) >= swing_strength and np.sum(high[i] > rh) >= swing_strength:
                ref_low = low[max(0, i-w*2):i].min() if i > 0 else low[i]
                sp = (high[i] - ref_low) / (ref_low + 1e-10) * 100
                if sp >= min_swing_pct and (high[i] - ref_low) > atr[i] * flat_zone_pct:
                    labels[i] = 1; continue
        ll = low[max(0, i-w):i]; rl = low[i+1:min(n, i+w+1)]
        if len(ll) >= swing_strength and len(rl) >= swing_strength:
            if np.sum(low[i] < ll) >= swing_strength and np.sum(low[i] < rl) >= swing_strength:
                ref_high = high[max(0, i-w*2):i].max() if i > 0 else high[i]
                sp = (ref_high - low[i]) / (ref_high + 1e-10) * 100
                if sp >= min_swing_pct and (ref_high - low[i]) > atr[i] * flat_zone_pct:
                    labels[i] = -1
    df = df.copy()
    df["label"] = labels
    return df


def run(
    symbol: str,
    processed_dir: str,
    pysr_dir: str,
    features_dir: str,
    base_interval: str = "1h",
    label_interval: str = "4h",
    extra_intervals: List[str] = None,
    swing_window: int = 5,
    swing_strength: int = 2,
    min_swing_pct: float = 0.3,
    flat_zone_pct: float = 0.1,
    skip: bool = False,
):
    if skip:
        log.info("[Step5] skip=True, skipping feature building.")
        return

    if extra_intervals is None:
        extra_intervals = [i for i in ["5m", "15m", "1h", "4h", "1d"] if i != base_interval]

    proc_path = Path(processed_dir)
    pysr_path = Path(pysr_dir)
    feat_path = Path(features_dir)
    feat_path.mkdir(parents=True, exist_ok=True)

    # Load base timeframe (label interval = 4h)
    label_path = proc_path / "klines" / symbol / f"{symbol}_{label_interval}.parquet"
    if not label_path.exists():
        log.error(f"Label interval klines not found: {label_path}")
        sys.exit(1)

    base_df = pd.read_parquet(label_path).sort_values("open_time").reset_index(drop=True)
    log.info(f"[Step5] Base df: {len(base_df)} bars @ {label_interval}")

    # Apply labels
    base_df = label_swings(base_df, swing_window, swing_strength, min_swing_pct, flat_zone_pct)
    log.info(f"[Step5] Labels: {base_df['label'].value_counts().to_dict()}")

    # Build features
    log.info("[Step5] Building label-interval features...")
    base_features = build_kline_features(base_df, prefix="lbl")

    log.info("[Step5] Building MTF features...")
    mtf_features = merge_mtf_features(base_df, proc_path, symbol, extra_intervals)

    log.info("[Step5] Merging aggTrade features...")
    at_features = merge_aggtrades_features(base_df, proc_path, symbol)

    log.info("[Step5] Merging funding rate features...")
    fr_features = merge_funding_features(base_df, proc_path, symbol)

    log.info("[Step5] Evaluating PySR features...")
    pysr_features = merge_pysr_features(base_df, base_features, pysr_path)

    # Combine all
    all_parts = [base_features, mtf_features]
    if not at_features.empty: all_parts.append(at_features)
    if not fr_features.empty: all_parts.append(fr_features)
    if not pysr_features.empty: all_parts.append(pysr_features)

    combined = pd.concat(all_parts, axis=1)
    combined.insert(0, "open_time", base_df["open_time"].values)
    combined["label"] = base_df["label"].values
    combined["close"] = base_df["close"].values
    combined["high"] = base_df["high"].values
    combined["low"] = base_df["low"].values

    # Remove duplicated columns
    combined = combined.loc[:, ~combined.columns.duplicated()]

    n_features = combined.shape[1] - 5  # subtract meta cols
    log.info(f"[Step5] Total features: {n_features} | Total rows: {len(combined)}")

    if n_features < 400:
        log.warning(f"[Step5] Feature count {n_features} below target 400. Check MTF availability.")

    out_path = feat_path / symbol / f"{symbol}_{label_interval}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out_path, index=False)
    log.info(f"[Step5] Saved features: {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step5: Feature Engineering")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--pysr_dir", default="data/pysr")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--base_interval", default="1h")
    parser.add_argument("--label_interval", default="4h")
    parser.add_argument("--extra_intervals", nargs="+", default=["5m", "15m", "1h", "1d"])
    parser.add_argument("--swing_window", type=int, default=5)
    parser.add_argument("--swing_strength", type=int, default=2)
    parser.add_argument("--min_swing_pct", type=float, default=0.3)
    parser.add_argument("--flat_zone_pct", type=float, default=0.1)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    run(
        symbol=args.symbol,
        processed_dir=args.processed_dir,
        pysr_dir=args.pysr_dir,
        features_dir=args.features_dir,
        base_interval=args.base_interval,
        label_interval=args.label_interval,
        extra_intervals=args.extra_intervals,
        swing_window=args.swing_window,
        swing_strength=args.swing_strength,
        min_swing_pct=args.min_swing_pct,
        flat_zone_pct=args.flat_zone_pct,
        skip=args.skip,
    )
