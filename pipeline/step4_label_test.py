"""Step 4 - Label Tester (STANDALONE, not part of pipeline).

Test different Swing High/Low labeling parameters and visualize samples.
Run this independently to find the best label parameters before training.

Usage:
    python pipeline/step4_label_test.py \
        --processed_dir data/processed \
        --symbol BTCUSDT \
        --interval 4h \
        --swing_window 5 \
        --swing_strength 2 \
        --min_swing_pct 0.3 \
        --flat_zone_pct 0.1 \
        --output_html reports/label_test.html
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def label_swings(
    df: pd.DataFrame,
    swing_window: int = 5,
    swing_strength: int = 2,
    min_swing_pct: float = 0.3,
    flat_zone_pct: float = 0.1,
) -> pd.DataFrame:
    """Label each bar as swing high (+1), swing low (-1), or flat (0).

    Args:
        df: OHLCV DataFrame with 'high', 'low', 'close' columns.
        swing_window: Number of bars on each side to check for swing confirmation.
        swing_strength: How many consecutive higher highs/lower lows needed.
        min_swing_pct: Minimum price move % to qualify as a swing.
        flat_zone_pct: ATR fraction below which a swing is classified as flat.

    Returns:
        DataFrame with added 'label' column: 1=swing_high, -1=swing_low, 0=flat.
    """
    df = df.copy()
    n = len(df)
    labels = np.zeros(n, dtype=int)
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values

    # ATR for flat zone filtering
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1)))
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    w = swing_window
    for i in range(w, n - w):
        # Swing High check
        left_high = high[max(0, i - w):i]
        right_high = high[i + 1:min(n, i + w + 1)]
        if len(left_high) >= swing_strength and len(right_high) >= swing_strength:
            is_swing_high = (
                np.sum(high[i] > left_high) >= swing_strength and
                np.sum(high[i] > right_high) >= swing_strength
            )
            if is_swing_high:
                # Check min swing size
                ref_low = low[max(0, i - w * 2):i].min() if i > 0 else low[i]
                swing_pct = (high[i] - ref_low) / (ref_low + 1e-10) * 100
                flat_threshold = atr[i] * flat_zone_pct
                if swing_pct >= min_swing_pct and (high[i] - ref_low) > flat_threshold:
                    labels[i] = 1
                    continue

        # Swing Low check
        left_low = low[max(0, i - w):i]
        right_low = low[i + 1:min(n, i + w + 1)]
        if len(left_low) >= swing_strength and len(right_low) >= swing_strength:
            is_swing_low = (
                np.sum(low[i] < left_low) >= swing_strength and
                np.sum(low[i] < right_low) >= swing_strength
            )
            if is_swing_low:
                ref_high = high[max(0, i - w * 2):i].max() if i > 0 else high[i]
                swing_pct = (ref_high - low[i]) / (ref_high + 1e-10) * 100
                flat_threshold = atr[i] * flat_zone_pct
                if swing_pct >= min_swing_pct and (ref_high - low[i]) > flat_threshold:
                    labels[i] = -1

    df["label"] = labels
    return df


def compute_label_stats(df: pd.DataFrame) -> dict:
    """Compute statistics about the label distribution."""
    counts = df["label"].value_counts().to_dict()
    total = len(df)
    stats = {
        "total_bars": total,
        "swing_high_count": counts.get(1, 0),
        "swing_low_count": counts.get(-1, 0),
        "flat_count": counts.get(0, 0),
        "swing_high_pct": round(counts.get(1, 0) / total * 100, 2),
        "swing_low_pct": round(counts.get(-1, 0) / total * 100, 2),
        "flat_pct": round(counts.get(0, 0) / total * 100, 2),
    }

    # Avg bars between swing points
    swings = df[df["label"] != 0].index.tolist()
    if len(swings) > 1:
        gaps = [swings[i+1] - swings[i] for i in range(len(swings)-1)]
        stats["avg_bars_between_swings"] = round(np.mean(gaps), 1)
        stats["median_bars_between_swings"] = round(np.median(gaps), 1)
    else:
        stats["avg_bars_between_swings"] = None
        stats["median_bars_between_swings"] = None

    # Hold time analysis (for reversal timing)
    high_low_alternating = 0
    prev_label = 0
    for lbl in df[df["label"] != 0]["label"]:
        if prev_label != 0 and lbl != prev_label:
            high_low_alternating += 1
        prev_label = lbl
    stats["alternating_ratio"] = round(high_low_alternating / max(len(swings)-1, 1), 3)

    return stats


def build_html_report(
    df: pd.DataFrame,
    stats: dict,
    params: dict,
    symbol: str,
    interval: str,
    output_path: Path,
    max_candles_display: int = 500,
):
    """Build interactive HTML report with Plotly."""
    # Sample a window for display
    display_df = df.tail(max_candles_display).copy()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=[f"{symbol} {interval} - Swing Labels", "Volume", "Label Distribution"],
        vertical_spacing=0.05,
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=display_df["open_time"],
        open=display_df["open"],
        high=display_df["high"],
        low=display_df["low"],
        close=display_df["close"],
        name="OHLC",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Swing Highs
    sh = display_df[display_df["label"] == 1]
    fig.add_trace(go.Scatter(
        x=sh["open_time"], y=sh["high"] * 1.001,
        mode="markers",
        marker=dict(symbol="triangle-down", size=12, color="#ef5350"),
        name="Swing High",
    ), row=1, col=1)

    # Swing Lows
    sl = display_df[display_df["label"] == -1]
    fig.add_trace(go.Scatter(
        x=sl["open_time"], y=sl["low"] * 0.999,
        mode="markers",
        marker=dict(symbol="triangle-up", size=12, color="#26a69a"),
        name="Swing Low",
    ), row=1, col=1)

    # Volume
    colors = ["#26a69a" if c >= o else "#ef5350"
              for c, o in zip(display_df["close"], display_df["open"])]
    fig.add_trace(go.Bar(
        x=display_df["open_time"],
        y=display_df["volume"],
        marker_color=colors,
        name="Volume",
        opacity=0.7,
    ), row=2, col=1)

    # Label rolling distribution
    label_series = display_df["label"].replace({1: 1, -1: -1, 0: 0})
    fig.add_trace(go.Scatter(
        x=display_df["open_time"],
        y=label_series.rolling(20).mean(),
        mode="lines",
        name="Label MA(20)",
        line=dict(color="#ff9800", width=2),
    ), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    fig.update_layout(
        height=900,
        title=f"Label Test | {symbol} {interval} | window={params['swing_window']} strength={params['swing_strength']} min_pct={params['min_swing_pct']}%",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        showlegend=True,
        font=dict(family="Inter, sans-serif", size=12),
    )

    # Build HTML
    chart_html = fig.to_html(full_html=False, include_plotlyjs=True)

    stats_rows = "".join(
        f'<tr><td>{k}</td><td><strong>{v}</strong></td></tr>'
        for k, v in stats.items()
    )

    param_rows = "".join(
        f'<tr><td>{k}</td><td><strong>{v}</strong></td></tr>'
        for k, v in params.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Label Test - {symbol} {interval}</title>
<style>
body {{ font-family: Inter, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; margin: 0; }}
.container {{ max-width: 1400px; margin: 0 auto; }}
h1 {{ color: #00d4aa; margin-bottom: 8px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
.card {{ background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 16px; }}
.card h3 {{ margin: 0 0 12px 0; color: #00d4aa; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
td {{ padding: 6px 8px; border-bottom: 1px solid #0f3460; }}
td:first-child {{ color: #888; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
.badge-green {{ background: #1a3a2a; color: #26a69a; }}
.badge-red {{ background: #3a1a1a; color: #ef5350; }}
.badge-gray {{ background: #2a2a2a; color: #888; }}
</style>
</head>
<body>
<div class="container">
<h1>ZongBot - Label Tester</h1>
<p style="color:#888">{symbol} {interval} | Total bars: {stats['total_bars']} | Showing last {min(max_candles_display, stats['total_bars'])} candles</p>
<div class="grid">
<div class="card">
<h3>Label Statistics</h3>
<table>{stats_rows}</table>
</div>
<div class="card">
<h3>Parameters Used</h3>
<table>{param_rows}</table>
<br>
<p style="color:#888; font-size:12px">Tip: Increase swing_window for longer-term swings. Increase min_swing_pct to filter noise. Aim for alternating_ratio > 0.7.</p>
</div>
</div>
{chart_html}
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Label test report saved: {output_path}")


def run(
    processed_dir: str,
    symbol: str,
    interval: str,
    swing_window: int,
    swing_strength: int,
    min_swing_pct: float,
    flat_zone_pct: float,
    output_html: str,
):
    proc_path = Path(processed_dir)
    kline_path = proc_path / "klines" / symbol / f"{symbol}_{interval}.parquet"

    if not kline_path.exists():
        log.error(f"Klines file not found: {kline_path}")
        sys.exit(1)

    df = pd.read_parquet(kline_path)
    df = df.sort_values("open_time").reset_index(drop=True)
    log.info(f"Loaded {len(df)} bars for {symbol} {interval}")

    params = dict(
        swing_window=swing_window,
        swing_strength=swing_strength,
        min_swing_pct=min_swing_pct,
        flat_zone_pct=flat_zone_pct,
    )

    df = label_swings(df, **params)
    stats = compute_label_stats(df)

    log.info("Label distribution:")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    build_html_report(
        df=df,
        stats=stats,
        params=params,
        symbol=symbol,
        interval=interval,
        output_path=Path(output_html),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step4: Label Tester (standalone)")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="4h")
    parser.add_argument("--swing_window", type=int, default=5)
    parser.add_argument("--swing_strength", type=int, default=2)
    parser.add_argument("--min_swing_pct", type=float, default=0.3)
    parser.add_argument("--flat_zone_pct", type=float, default=0.1)
    parser.add_argument("--output_html", default="reports/label_test.html")
    args = parser.parse_args()
    run(
        processed_dir=args.processed_dir,
        symbol=args.symbol,
        interval=args.interval,
        swing_window=args.swing_window,
        swing_strength=args.swing_strength,
        min_swing_pct=args.min_swing_pct,
        flat_zone_pct=args.flat_zone_pct,
        output_html=args.output_html,
    )
