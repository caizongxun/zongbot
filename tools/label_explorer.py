"""
Label Explorer (Standalone Tool - NOT part of pipeline)
=========================================================
Visually inspect how different swing-label parameters affect
the training samples seen by the model.

Usage:
    python tools/label_explorer.py \\
        --parquet data/processed/BTCUSDT_1h.parquet \\
        --left 5 --right 5 --min_move 0.8 \\
        --atr_mult 0 --merge 3 \\
        --start 2023-01-01 --end 2023-06-30 \\
        --output label_report.html

Outputs a self-contained HTML file with:
  - Candlestick chart with swing labels overlaid
  - Label distribution table
  - Per-label price statistics
  - Forward-return distributions at swing points
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import json

# Allow running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.labels.label_engine import LabelEngine, LabelConfig


def parse_args():
    p = argparse.ArgumentParser(description="BTC Label Explorer")
    p.add_argument("--parquet", required=True, help="Path to processed OHLCV parquet file")
    p.add_argument("--left", type=int, default=5, help="Left bars for swing detection")
    p.add_argument("--right", type=int, default=5, help="Right bars for swing detection")
    p.add_argument("--min_move", type=float, default=0.5, help="Min move %% to qualify swing")
    p.add_argument("--atr_mult", type=float, default=0.0, help="ATR multiplier (0 = use min_move)")
    p.add_argument("--atr_period", type=int, default=14)
    p.add_argument("--merge", type=int, default=3, help="Merge window bars")
    p.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    p.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    p.add_argument("--output", default="label_report.html", help="Output HTML filename")
    p.add_argument("--forward_bars", type=int, default=20, help="Forward bars for return analysis")
    return p.parse_args()


def compute_forward_returns(df: pd.DataFrame, forward_bars: int) -> pd.DataFrame:
    df["fwd_return"] = (df["close"].shift(-forward_bars) / df["close"] - 1) * 100
    return df


def build_html(df: pd.DataFrame, stats: dict, args) -> str:
    # Prepare chart data (downsample if > 2000 rows to keep HTML small)
    chart_df = df.copy()
    if len(chart_df) > 2000:
        chart_df = chart_df.iloc[::max(1, len(chart_df)//2000)]

    times = chart_df.index.strftime("%Y-%m-%d %H:%M").tolist()
    opens = chart_df["open"].tolist()
    highs = chart_df["high"].tolist()
    lows = chart_df["low"].tolist()
    closes = chart_df["close"].tolist()

    sh_idx = df[df["label"] == 1].index
    sl_idx = df[df["label"] == -1].index

    sh_times = sh_idx.strftime("%Y-%m-%d %H:%M").tolist()
    sh_prices = df.loc[sh_idx, "swing_price"].tolist()
    sl_times = sl_idx.strftime("%Y-%m-%d %H:%M").tolist()
    sl_prices = df.loc[sl_idx, "swing_price"].tolist()

    # Forward return stats
    sh_fwd = df.loc[sh_idx, "fwd_return"].dropna().tolist()
    sl_fwd = df.loc[sl_idx, "fwd_return"].dropna().tolist()

    params_str = (
        f"left={args.left}, right={args.right}, "
        f"min_move={args.min_move}%, atr_mult={args.atr_mult}, "
        f"merge={args.merge}"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ZongBot Label Explorer</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  :root {{
    --bg: #0e0f12; --surface: #16181d; --surface2: #1e2028;
    --border: #2a2d36; --text: #d0d4e0; --muted: #7a7f94;
    --primary: #4f98a3; --green: #3dba7a; --red: #e05c6e;
    --font: 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: var(--font); font-size: 14px; }}
  header {{ background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 14px 24px; display: flex; align-items: center; gap: 16px; }}
  .logo {{ font-size: 20px; font-weight: 700; color: var(--primary); letter-spacing: -0.5px; }}
  .params {{ color: var(--muted); font-size: 12px; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 320px; gap: 20px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }}
  .card h3 {{ font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px;
    margin-bottom: 12px; font-weight: 600; }}
  #chart-main {{ height: 520px; }}
  .stat-row {{ display: flex; justify-content: space-between; align-items: center;
    padding: 8px 0; border-bottom: 1px solid var(--border); }}
  .stat-row:last-child {{ border-bottom: none; }}
  .stat-label {{ color: var(--muted); font-size: 12px; }}
  .stat-value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
  .tag-high {{ color: var(--red); }} .tag-low {{ color: var(--green); }}
  .dist-chart {{ height: 200px; margin-top: 8px; }}
  .section-title {{ font-size: 16px; font-weight: 700; margin: 24px 0 12px; }}
  .samples-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  .samples-table th {{ background: var(--surface2); color: var(--muted); padding: 8px 12px;
    text-align: left; border-bottom: 1px solid var(--border); }}
  .samples-table td {{ padding: 7px 12px; border-bottom: 1px solid var(--border); }}
  .samples-table tr:hover td {{ background: var(--surface2); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: 600; }}
  .badge-high {{ background: rgba(224,92,110,0.18); color: var(--red); }}
  .badge-low {{ background: rgba(61,186,122,0.18); color: var(--green); }}
  .badge-flat {{ background: rgba(122,127,148,0.18); color: var(--muted); }}
</style>
</head>
<body>
<header>
  <div class="logo">ZongBot</div>
  <div style="flex:1;"><span style="color:var(--text);font-weight:600;">Label Explorer</span></div>
  <div class="params">Params: {params_str}</div>
</header>
<div class="container">
  <div class="grid">
    <div>
      <div class="card">
        <h3>BTC Price Chart + Swing Labels</h3>
        <div id="chart-main"></div>
      </div>
      <div class="card" style="margin-top:20px;">
        <h3>Forward Return Distribution at Swing Points ({args.forward_bars} bars)</h3>
        <div id="dist-chart" class="dist-chart" style="height:280px;"></div>
      </div>
    </div>
    <div>
      <div class="card">
        <h3>Label Statistics</h3>
        <div class="stat-row">
          <span class="stat-label">Total Bars</span>
          <span class="stat-value">{stats["total_bars"]:,}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Swing Highs <span class="tag-high">(Long Exit)</span></span>
          <span class="stat-value tag-high">{stats["swing_high"]:,} ({stats["swing_high_pct"]}%)</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Swing Lows <span class="tag-low">(Long Entry)</span></span>
          <span class="stat-value tag-low">{stats["swing_low"]:,} ({stats["swing_low_pct"]}%)</span>
        </div>
        <div class="stat-row">
          <span class="stat-label">Flat (No Signal)</span>
          <span class="stat-value">{stats["flat"]:,}</span>
        </div>
      </div>
      <div class="card" style="margin-top:16px;">
        <h3>Forward Return Stats</h3>
        <div class="stat-row">
          <span class="stat-label tag-high">Swing High Mean Ret</span>
          <span class="stat-value">{f"{np.mean(sh_fwd):.2f}%" if sh_fwd else "N/A"}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label tag-high">Swing High Median</span>
          <span class="stat-value">{f"{np.median(sh_fwd):.2f}%" if sh_fwd else "N/A"}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label tag-low">Swing Low Mean Ret</span>
          <span class="stat-value">{f"{np.mean(sl_fwd):.2f}%" if sl_fwd else "N/A"}</span>
        </div>
        <div class="stat-row">
          <span class="stat-label tag-low">Swing Low Median</span>
          <span class="stat-value">{f"{np.median(sl_fwd):.2f}%" if sl_fwd else "N/A"}</span>
        </div>
      </div>
      <div class="card" style="margin-top:16px;">
        <h3>Parameters</h3>
        <div class="stat-row"><span class="stat-label">Left Bars</span><span class="stat-value">{args.left}</span></div>
        <div class="stat-row"><span class="stat-label">Right Bars</span><span class="stat-value">{args.right}</span></div>
        <div class="stat-row"><span class="stat-label">Min Move %</span><span class="stat-value">{args.min_move}</span></div>
        <div class="stat-row"><span class="stat-label">ATR Multiplier</span><span class="stat-value">{args.atr_mult}</span></div>
        <div class="stat-row"><span class="stat-label">Merge Window</span><span class="stat-value">{args.merge}</span></div>
        <div class="stat-row"><span class="stat-label">Forward Bars</span><span class="stat-value">{args.forward_bars}</span></div>
      </div>
    </div>
  </div>

  <div class="section-title">Sample Swing Events (first 200)</div>
  <div class="card">
    <table class="samples-table">
      <thead>
        <tr><th>Timestamp</th><th>Type</th><th>Pivot Price</th><th>Fwd Return ({args.forward_bars}b)</th><th>Volume</th></tr>
      </thead>
      <tbody id="samples-body"></tbody>
    </table>
  </div>
</div>

<script>
const times = {json.dumps(times)};
const opens = {json.dumps([round(x,2) for x in opens])};
const highs_data = {json.dumps([round(x,2) for x in highs])};
const lows_data = {json.dumps([round(x,2) for x in lows])};
const closes = {json.dumps([round(x,2) for x in closes])};
const sh_times = {json.dumps(sh_times)};
const sh_prices = {json.dumps([round(x,2) for x in sh_prices])};
const sl_times = {json.dumps(sl_times)};
const sl_prices = {json.dumps([round(x,2) for x in sl_prices])};
const sh_fwd = {json.dumps([round(x,3) for x in sh_fwd])};
const sl_fwd = {json.dumps([round(x,3) for x in sl_fwd])};

const candlestick = {{
  type: 'candlestick', x: times,
  open: opens, high: highs_data, low: lows_data, close: closes,
  name: 'BTC',
  increasing: {{line: {{color: '#3dba7a'}}}},
  decreasing: {{line: {{color: '#e05c6e'}}}},
}};
const sh_scatter = {{
  type: 'scatter', mode: 'markers', x: sh_times, y: sh_prices,
  marker: {{symbol: 'triangle-down', size: 12, color: '#e05c6e'}},
  name: 'Swing High'
}};
const sl_scatter = {{
  type: 'scatter', mode: 'markers', x: sl_times, y: sl_prices,
  marker: {{symbol: 'triangle-up', size: 12, color: '#3dba7a'}},
  name: 'Swing Low'
}};

const layout = {{
  paper_bgcolor: '#16181d', plot_bgcolor: '#16181d',
  font: {{color: '#d0d4e0', family: 'Segoe UI, system-ui, sans-serif', size: 12}},
  xaxis: {{gridcolor: '#2a2d36', showgrid: true, rangeslider: {{visible: false}}}},
  yaxis: {{gridcolor: '#2a2d36', showgrid: true}},
  legend: {{bgcolor: 'rgba(0,0,0,0)', bordercolor: '#2a2d36', borderwidth: 1}},
  margin: {{l: 60, r: 20, t: 20, b: 40}},
}};
Plotly.newPlot('chart-main', [candlestick, sh_scatter, sl_scatter], layout, {{responsive:true}});

// Forward return distribution
const dist_layout = {{
  paper_bgcolor: '#16181d', plot_bgcolor: '#16181d',
  font: {{color: '#d0d4e0', size: 11}},
  xaxis: {{title: 'Forward Return %', gridcolor: '#2a2d36'}},
  yaxis: {{title: 'Count', gridcolor: '#2a2d36'}},
  barmode: 'overlay',
  margin: {{l: 50, r: 20, t: 10, b: 50}},
  legend: {{bgcolor: 'rgba(0,0,0,0)'}},
}};
Plotly.newPlot('dist-chart', [
  {{type:'histogram', x: sh_fwd, name:'Swing High Fwd', marker:{{color:'rgba(224,92,110,0.7)'}}, opacity:0.75, nbinsx:40}},
  {{type:'histogram', x: sl_fwd, name:'Swing Low Fwd', marker:{{color:'rgba(61,186,122,0.7)'}}, opacity:0.75, nbinsx:40}},
], dist_layout, {{responsive:true}});

// Sample table
const samples_data = {json.dumps([
    {
        "time": str(idx),
        "type": row["swing_type"],
        "price": round(float(row["swing_price"]), 2) if not pd.isna(row["swing_price"]) else None,
        "fwd": round(float(row["fwd_return"]), 3) if "fwd_return" in df.columns and not pd.isna(row.get("fwd_return", float("nan"))) else None,
        "volume": round(float(row["volume"]), 2),
    }
    for idx, row in df[df["label"] != 0].head(200).iterrows()
])};
const tbody = document.getElementById('samples-body');
samples_data.forEach(s => {{
  const cls = s.type === 'swing_high' ? 'high' : 'low';
  tbody.innerHTML += `<tr>
    <td>${{s.time}}</td>
    <td><span class="badge badge-${{cls}}">${{s.type.replace('_',' ').toUpperCase()}}</span></td>
    <td style="font-variant-numeric:tabular-nums">${{s.price ?? 'N/A'}}</td>
    <td style="color:${{s.fwd > 0 ? '#3dba7a' : '#e05c6e'}}; font-variant-numeric:tabular-nums">${{s.fwd !== null ? s.fwd.toFixed(2)+'%' : 'N/A'}}</td>
    <td style="font-variant-numeric:tabular-nums">${{s.volume?.toLocaleString() ?? ''}}</td>
  </tr>`;
}});
</script>
</body>
</html>"""
    return html


def main():
    args = parse_args()

    print(f"Loading data from: {args.parquet}")
    df = pd.read_parquet(args.parquet)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "open_time" in df.columns:
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce").fillna(
                pd.to_datetime(df["open_time"], errors="coerce")
            )
            df = df.set_index("open_time")
        elif "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"]))

    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index <= args.end]

    print(f"Data shape: {df.shape}, date range: {df.index[0]} -> {df.index[-1]}")

    cfg = LabelConfig(
        left_bars=args.left,
        right_bars=args.right,
        min_move_pct=args.min_move,
        atr_multiplier=args.atr_mult,
        atr_period=args.atr_period,
        merge_window=args.merge,
    )
    engine = LabelEngine(cfg)
    labeled = engine.fit_transform(df)
    labeled = compute_forward_returns(labeled, args.forward_bars)
    stats = LabelEngine.label_stats(labeled)

    print("\n=== Label Statistics ===")
    for k, v in stats.items():
        print(f"  {k:20s}: {v}")

    html = build_html(labeled, stats, args)
    out_path = args.output
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nReport saved to: {out_path}")
    print("Open in browser to explore labels visually.")


if __name__ == "__main__":
    main()
