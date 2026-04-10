"""
Step 4 - Label Engine
=====================
Swing High / Low label detection for BTC reversal prediction.

Label convention:
  1  = Swing High (potential SHORT entry zone / long exit)
 -1  = Swing Low  (potential LONG  entry zone / short exit)
  0  = Flat / no clear swing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class LabelConfig:
    """Parameters that control swing detection."""
    left_bars: int = 5          # bars to the left that must be lower/higher
    right_bars: int = 5         # bars to the right that must be lower/higher
    min_move_pct: float = 0.5   # minimum price move % to qualify as swing
    atr_multiplier: float = 0.0 # if > 0, use ATR-based min move instead of pct
    atr_period: int = 14
    merge_window: int = 3       # merge swings within N bars of each other
    label_mode: Literal["binary", "ternary"] = "ternary"
    # ternary => {-1, 0, 1}; binary => map -1->0, 1->1 (for single model)


class LabelEngine:
    """Detects swing pivots and assigns labels to a OHLCV DataFrame."""

    def __init__(self, cfg: LabelConfig | None = None):
        self.cfg = cfg or LabelConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: open, high, low, close, volume.
            Index should be a DatetimeIndex.

        Returns
        -------
        pd.DataFrame
            Original df with extra columns:
              label        : int  (-1, 0, 1)
              swing_type   : str  ('swing_high', 'swing_low', 'flat')
              swing_price  : float (pivot price or NaN)
        """
        df = df.copy()
        df = self._ensure_columns(df)
        df = self._compute_atr(df)
        df["label"] = 0
        df["swing_type"] = "flat"
        df["swing_price"] = np.nan

        highs = df["high"].values
        lows = df["low"].values
        closes = df["close"].values
        atrs = df["_atr"].values
        n = len(df)
        lb = self.cfg.left_bars
        rb = self.cfg.right_bars

        for i in range(lb, n - rb):
            pivot_h = highs[i]
            pivot_l = lows[i]

            # Swing High: pivot high is the maximum in [i-lb, i+rb]
            window_h = highs[i - lb: i + rb + 1]
            if pivot_h == window_h.max():
                move = self._min_move(pivot_h, closes[i - lb], atrs[i])
                if move:
                    df.iat[i, df.columns.get_loc("label")] = 1
                    df.iat[i, df.columns.get_loc("swing_type")] = "swing_high"
                    df.iat[i, df.columns.get_loc("swing_price")] = pivot_h

            # Swing Low: pivot low is the minimum in [i-lb, i+rb]
            window_l = lows[i - lb: i + rb + 1]
            if pivot_l == window_l.min():
                move = self._min_move(closes[i - lb], pivot_l, atrs[i])
                if move:
                    df.iat[i, df.columns.get_loc("label")] = -1
                    df.iat[i, df.columns.get_loc("swing_type")] = "swing_low"
                    df.iat[i, df.columns.get_loc("swing_price")] = pivot_l

        # Resolve conflicts (both high and low flagged on same bar -> use stronger move)
        conflict_mask = (df["label"].shift(0) != 0) & (df["swing_type"] == "flat")
        # merge close swings
        df = self._merge_close_swings(df)

        if self.cfg.label_mode == "binary":
            df["label"] = df["label"].clip(lower=0)  # -1 -> 0, 1 -> 1

        # Drop helper columns
        df.drop(columns=["_atr"], errors="ignore", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns.str.lower())
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        df.columns = df.columns.str.lower()
        return df

    def _compute_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.cfg.atr_period
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ], axis=1).max(axis=1)
        df["_atr"] = tr.rolling(period, min_periods=1).mean()
        return df

    def _min_move(self, price_high: float, price_low: float, atr: float) -> bool:
        """Return True if the move qualifies as significant."""
        if price_high <= 0:
            return False
        pct_move = abs(price_high - price_low) / price_high * 100
        if self.cfg.atr_multiplier > 0:
            required = self.cfg.atr_multiplier * atr / price_high * 100
        else:
            required = self.cfg.min_move_pct
        return pct_move >= required

    def _merge_close_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Within merge_window bars, keep only the first swing."""
        w = self.cfg.merge_window
        if w <= 1:
            return df
        label_col = df.columns.get_loc("label")
        labels = df["label"].values.copy()
        last_swing_idx = -w - 1
        last_swing_type = 0
        for i in range(len(labels)):
            if labels[i] != 0:
                if i - last_swing_idx <= w and labels[i] == last_swing_type:
                    labels[i] = 0
                    df.iat[i, df.columns.get_loc("swing_type")] = "flat"
                    df.iat[i, df.columns.get_loc("swing_price")] = np.nan
                else:
                    last_swing_idx = i
                    last_swing_type = labels[i]
        df.iloc[:, label_col] = labels
        return df

    # ------------------------------------------------------------------
    # Statistics helper
    # ------------------------------------------------------------------

    @staticmethod
    def label_stats(df: pd.DataFrame) -> dict:
        counts = df["label"].value_counts().to_dict()
        total = len(df)
        return {
            "total_bars": total,
            "swing_high": counts.get(1, 0),
            "swing_low": counts.get(-1, 0),
            "flat": counts.get(0, total),
            "swing_high_pct": round(counts.get(1, 0) / total * 100, 2),
            "swing_low_pct": round(counts.get(-1, 0) / total * 100, 2),
        }


def generate_labels(
    df: pd.DataFrame,
    left_bars: int = 5,
    right_bars: int = 5,
    min_move_pct: float = 0.5,
    atr_multiplier: float = 0.0,
    atr_period: int = 14,
    merge_window: int = 3,
    label_mode: str = "ternary",
) -> pd.DataFrame:
    """Convenience wrapper around LabelEngine."""
    cfg = LabelConfig(
        left_bars=left_bars,
        right_bars=right_bars,
        min_move_pct=min_move_pct,
        atr_multiplier=atr_multiplier,
        atr_period=atr_period,
        merge_window=merge_window,
        label_mode=label_mode,  # type: ignore
    )
    engine = LabelEngine(cfg)
    return engine.fit_transform(df)
