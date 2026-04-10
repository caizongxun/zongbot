"""Step 7: OOS Walk-Forward Backtester.

Bar-by-bar simulation to prevent future data leakage.
- Compound position sizing: 15-20% of equity per trade
- 5x leverage
- Strict stop-loss based on ATR
- Fee + slippage modeling
- Starting capital: 100 USDT
- Outputs interactive HTML report + JSON stats

Usage:
    python pipeline/step7_backtest.py \
        --version v2025.Q1.0 \
        --symbol BTCUSDT \
        --models_dir models \
        --features_dir data/features \
        --output_dir reports \
        --label_interval 4h \
        --initial_capital 100 \
        --risk_per_trade 0.15 \
        --leverage 5 \
        --sl_atr_mult 1.5 \
        --fee_rate 0.0004 \
        --slippage_rate 0.0002 \
        --long_threshold 0.55 \
        --short_threshold 0.55
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


class Position:
    def __init__(self, direction, entry_price, size_usdt, leverage, stop_loss, entry_time):
        self.direction = direction  # "long" or "short"
        self.entry_price = entry_price
        self.size_usdt = size_usdt  # notional margin
        self.leverage = leverage
        self.stop_loss = stop_loss
        self.entry_time = entry_time
        self.contracts = (size_usdt * leverage) / entry_price

    def pnl(self, current_price: float) -> float:
        if self.direction == "long":
            return (current_price - self.entry_price) * self.contracts
        else:
            return (self.entry_price - current_price) * self.contracts

    def pnl_pct(self, current_price: float) -> float:
        return self.pnl(current_price) / (self.size_usdt + 1e-10)

    def is_stopped(self, bar_low: float, bar_high: float) -> bool:
        if self.direction == "long":
            return bar_low <= self.stop_loss
        else:
            return bar_high >= self.stop_loss


class Backtester:
    def __init__(
        self,
        initial_capital: float = 100.0,
        risk_per_trade: float = 0.15,
        leverage: int = 5,
        sl_atr_mult: float = 1.5,
        fee_rate: float = 0.0004,
        slippage_rate: float = 0.0002,
        long_threshold: float = 0.55,
        short_threshold: float = 0.55,
        max_hold_bars: int = 48,  # Force exit after N bars
    ):
        self.initial_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.leverage = leverage
        self.sl_atr_mult = sl_atr_mult
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.max_hold_bars = max_hold_bars

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        h = df["high"]; lo = df["low"]; c = df["close"]
        tr = pd.concat([
            h - lo,
            (h - c.shift(1)).abs(),
            (lo - c.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()

    def simulate(
        self,
        df: pd.DataFrame,
        long_model,
        short_model,
        feature_cols: list,
    ) -> dict:
        equity = self.initial_capital
        equity_curve = [equity]
        trades = []
        position: Optional[Position] = None
        bars_in_trade = 0

        atr = self.compute_atr(df)
        df = df.copy()
        df["atr"] = atr

        META_COLS = {"open_time", "label", "close", "high", "low", "atr"}
        feature_cols = [c for c in feature_cols if c not in META_COLS]

        log.info(f"[Step7] Simulating {len(df)} bars bar-by-bar...")

        for idx in range(len(df)):
            row = df.iloc[idx]
            bar_open = row["close"]  # use close as next bar approximation
            bar_high = row["high"]
            bar_low = row["low"]
            bar_close = row["close"]
            bar_time = row["open_time"]
            bar_atr = row["atr"] if not np.isnan(row["atr"]) else bar_close * 0.005

            # --- Check stop loss or max hold ---
            if position is not None:
                bars_in_trade += 1
                sl_hit = position.is_stopped(bar_low, bar_high)
                max_hold_hit = bars_in_trade >= self.max_hold_bars

                if sl_hit or max_hold_hit:
                    exit_price = position.stop_loss if sl_hit else bar_close
                    # Apply slippage
                    if position.direction == "long":
                        exit_price *= (1 - self.slippage_rate)
                    else:
                        exit_price *= (1 + self.slippage_rate)

                    gross_pnl = position.pnl(exit_price)
                    fee = (position.contracts * exit_price) * self.fee_rate
                    net_pnl = gross_pnl - fee
                    equity += net_pnl
                    equity = max(equity, 0.01)  # prevent negative

                    trades.append({
                        "entry_time": position.entry_time,
                        "exit_time": bar_time,
                        "direction": position.direction,
                        "entry_price": position.entry_price,
                        "exit_price": exit_price,
                        "stop_loss": position.stop_loss,
                        "size_usdt": position.size_usdt,
                        "gross_pnl": gross_pnl,
                        "fee": fee,
                        "net_pnl": net_pnl,
                        "pnl_pct": net_pnl / (position.size_usdt + 1e-10),
                        "exit_reason": "sl" if sl_hit else "max_hold",
                        "bars_held": bars_in_trade,
                        "equity_after": equity,
                    })
                    position = None
                    bars_in_trade = 0

            equity_curve.append(equity)

            # --- Signal generation (only if no open position) ---
            if position is None and idx >= 14:  # wait for ATR warmup
                try:
                    X_bar = df.iloc[[idx]][feature_cols].fillna(0)

                    long_prob = long_model.predict_proba(X_bar)[0][1]
                    short_prob = short_model.predict_proba(X_bar)[0][1]
                except Exception:
                    continue

                signal = None
                if long_prob >= self.long_threshold and long_prob > short_prob:
                    signal = "long"
                elif short_prob >= self.short_threshold and short_prob > long_prob:
                    signal = "short"

                if signal:
                    # Entry price with slippage
                    entry_price = bar_close * (1 + self.slippage_rate) if signal == "long" \
                        else bar_close * (1 - self.slippage_rate)

                    # Position size: % of equity, compound
                    size_usdt = equity * self.risk_per_trade
                    size_usdt = max(size_usdt, 1.0)  # min 1 USDT margin

                    # Stop loss: entry +/- ATR * multiplier
                    if signal == "long":
                        stop_loss = entry_price - bar_atr * self.sl_atr_mult
                    else:
                        stop_loss = entry_price + bar_atr * self.sl_atr_mult

                    # Entry fee
                    entry_notional = size_usdt * self.leverage
                    entry_fee = entry_notional / entry_price * entry_price * self.fee_rate
                    equity -= entry_fee

                    position = Position(
                        direction=signal,
                        entry_price=entry_price,
                        size_usdt=size_usdt,
                        leverage=self.leverage,
                        stop_loss=stop_loss,
                        entry_time=bar_time,
                    )
                    bars_in_trade = 0

        # Close any open position at end
        if position is not None:
            exit_price = df.iloc[-1]["close"]
            gross_pnl = position.pnl(exit_price)
            fee = (position.contracts * exit_price) * self.fee_rate
            net_pnl = gross_pnl - fee
            equity += net_pnl
            trades.append({
                "entry_time": position.entry_time,
                "exit_time": df.iloc[-1]["open_time"],
                "direction": position.direction,
                "entry_price": position.entry_price,
                "exit_price": exit_price,
                "stop_loss": position.stop_loss,
                "size_usdt": position.size_usdt,
                "gross_pnl": gross_pnl,
                "fee": fee,
                "net_pnl": net_pnl,
                "pnl_pct": net_pnl / (position.size_usdt + 1e-10),
                "exit_reason": "end_of_data",
                "bars_held": bars_in_trade,
                "equity_after": equity,
            })

        trades_df = pd.DataFrame(trades)
        return {
            "equity_curve": equity_curve,
            "trades": trades_df,
            "final_equity": equity,
            "total_return_pct": (equity - self.initial_capital) / self.initial_capital * 100,
        }


def compute_stats(result: dict, initial_capital: float) -> dict:
    trades = result["trades"]
    equity = result["equity_curve"]

    if len(trades) == 0:
        return {"error": "No trades taken"}

    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    
    # Drawdown
    eq_arr = np.array(equity)
    running_max = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - running_max) / (running_max + 1e-10) * 100
    max_dd = float(dd.min())

    # Sharpe (approximate)
    pnl_pcts = trades["pnl_pct"].values
    sharpe = float(np.mean(pnl_pcts) / (np.std(pnl_pcts) + 1e-10) * np.sqrt(252)) if len(pnl_pcts) > 1 else 0

    # Calmar
    total_ret = result["total_return_pct"]
    calmar = float(total_ret / (abs(max_dd) + 1e-10))

    stats = {
        "initial_capital": initial_capital,
        "final_equity": round(result["final_equity"], 4),
        "total_return_pct": round(total_ret, 2),
        "total_trades": len(trades),
        "long_trades": int((trades["direction"] == "long").sum()),
        "short_trades": int((trades["direction"] == "short").sum()),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "avg_win_pct": round(float(wins["pnl_pct"].mean() * 100) if len(wins) > 0 else 0, 3),
        "avg_loss_pct": round(float(losses["pnl_pct"].mean() * 100) if len(losses) > 0 else 0, 3),
        "profit_factor": round(float(wins["net_pnl"].sum() / (abs(losses["net_pnl"].sum()) + 1e-10)), 3),
        "max_drawdown_pct": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 3),
        "calmar_ratio": round(calmar, 3),
        "avg_bars_held": round(float(trades["bars_held"].mean()), 1),
        "total_fees": round(float(trades["fee"].sum()), 4),
        "sl_exits": int((trades["exit_reason"] == "sl").sum()),
        "max_hold_exits": int((trades["exit_reason"] == "max_hold").sum()),
    }
    return stats


def build_html_report(
    result: dict,
    stats: dict,
    version: str,
    symbol: str,
    oos_df: pd.DataFrame,
    output_path: Path,
):
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed, skipping HTML report.")
        return

    trades = result["trades"]
    equity_curve = result["equity_curve"]
    price_times = oos_df["open_time"].tolist()
    prices = oos_df["close"].tolist()

    # Equity curve time axis
    eq_times = price_times[:len(equity_curve)]
    if len(eq_times) < len(equity_curve):
        eq_times = eq_times + [price_times[-1]] * (len(equity_curve) - len(eq_times))

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.25, 0.2, 0.2],
        subplot_titles=["BTC Price + Trade Signals", "Equity Curve", "Per-Trade PnL %", "Drawdown %"],
        vertical_spacing=0.05,
    )

    # Price
    fig.add_trace(go.Scatter(x=price_times, y=prices, name="BTC Close",
                             line=dict(color="#aaa", width=1)), row=1, col=1)

    if len(trades) > 0:
        long_entries = trades[trades["direction"] == "long"]
        short_entries = trades[trades["direction"] == "short"]
        wins = trades[trades["net_pnl"] > 0]
        losses = trades[trades["net_pnl"] <= 0]

        fig.add_trace(go.Scatter(
            x=long_entries["entry_time"], y=long_entries["entry_price"],
            mode="markers", marker=dict(symbol="triangle-up", size=10, color="#26a69a"),
            name="Long Entry"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=short_entries["entry_time"], y=short_entries["entry_price"],
            mode="markers", marker=dict(symbol="triangle-down", size=10, color="#ef5350"),
            name="Short Entry"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=wins["exit_time"], y=wins["exit_price"],
            mode="markers", marker=dict(symbol="x", size=8, color="#66bb6a"),
            name="Win Exit"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=losses["exit_time"], y=losses["exit_price"],
            mode="markers", marker=dict(symbol="x", size=8, color="#ef9a9a"),
            name="Loss Exit"), row=1, col=1)

    # Equity curve
    fig.add_trace(go.Scatter(
        x=eq_times, y=equity_curve,
        fill="tozeroy", fillcolor="rgba(38,166,154,0.1)",
        line=dict(color="#26a69a", width=2), name="Equity"), row=2, col=1)

    # Per-trade PnL
    if len(trades) > 0:
        pnl_colors = ["#26a69a" if p > 0 else "#ef5350" for p in trades["net_pnl"]]
        fig.add_trace(go.Bar(
            x=trades["exit_time"], y=trades["pnl_pct"] * 100,
            marker_color=pnl_colors, name="Trade PnL%"), row=3, col=1)

    # Drawdown
    eq_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(eq_arr)
    dd = (eq_arr - running_max) / (running_max + 1e-10) * 100
    fig.add_trace(go.Scatter(
        x=eq_times, y=dd.tolist(),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
        line=dict(color="#ef5350", width=1.5), name="Drawdown%"), row=4, col=1)

    fig.update_layout(
        height=1100,
        title=f"ZongBot OOS Backtest | {symbol} | {version}",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        font=dict(family="Inter, sans-serif", size=12),
        legend=dict(orientation="h", y=-0.02),
    )

    chart_html = fig.to_html(full_html=False, include_plotlyjs=True)

    stats_rows = "".join(
        f'<tr><td>{k.replace("_", " ").title()}</td>'
        f'<td class="val {"pos" if isinstance(v, (int,float)) and v > 0 and "pct" in k.lower() else "neg" if isinstance(v, (int,float)) and v < 0 else ""}">'
        f'{v}</td></tr>'
        for k, v in stats.items()
    )

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>OOS Backtest - {symbol} {version}</title>
<style>
body {{ font-family: Inter, sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; margin: 0; }}
.container {{ max-width: 1600px; margin: 0 auto; }}
h1 {{ color: #00d4aa; }}
.stats-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; margin: 20px 0; }}
.stat-card {{ background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 14px 16px; }}
.stat-label {{ font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }}
.stat-value {{ font-size: 20px; font-weight: 700; color: #e0e0e0; }}
.stat-value.pos {{ color: #26a69a; }}
.stat-value.neg {{ color: #ef5350; }}
table {{ width: 100%; border-collapse: collapse; }}
td {{ padding: 5px 8px; border-bottom: 1px solid #0f3460; font-size: 13px; }}
td.val {{ text-align: right; font-weight: bold; }}
td.pos {{ color: #26a69a; }}
td.neg {{ color: #ef5350; }}
</style>
</head>
<body>
<div class="container">
<h1>ZongBot - OOS Backtest Report</h1>
<p style="color:#888">{symbol} | Version: {version} | Initial Capital: ${{stats.get('initial_capital', 100)}} | Leverage: 5x</p>
<div class="stats-grid">
  <div class="stat-card"><div class="stat-label">Final Equity</div><div class="stat-value {'pos' if stats.get('total_return_pct', 0) > 0 else 'neg'}">${'%.2f' % stats.get('final_equity', 0)}</div></div>
  <div class="stat-card"><div class="stat-label">Total Return</div><div class="stat-value {'pos' if stats.get('total_return_pct', 0) > 0 else 'neg'}">{'%.1f' % stats.get('total_return_pct', 0)}%</div></div>
  <div class="stat-card"><div class="stat-label">Win Rate</div><div class="stat-value">{'%.1f' % stats.get('win_rate', 0)}%</div></div>
  <div class="stat-card"><div class="stat-label">Max Drawdown</div><div class="stat-value neg">{'%.1f' % stats.get('max_drawdown_pct', 0)}%</div></div>
  <div class="stat-card"><div class="stat-label">Sharpe Ratio</div><div class="stat-value">{'%.3f' % stats.get('sharpe_ratio', 0)}</div></div>
  <div class="stat-card"><div class="stat-label">Total Trades</div><div class="stat-value">{stats.get('total_trades', 0)}</div></div>
</div>
{chart_html}
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"[Step7] Backtest report saved: {output_path}")


def run(
    version: str,
    symbol: str,
    models_dir: str,
    features_dir: str,
    output_dir: str,
    label_interval: str = "4h",
    initial_capital: float = 100.0,
    risk_per_trade: float = 0.15,
    leverage: int = 5,
    sl_atr_mult: float = 1.5,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0002,
    long_threshold: float = 0.55,
    short_threshold: float = 0.55,
    max_hold_bars: int = 48,
    skip: bool = False,
):
    if skip:
        log.info("[Step7] skip=True, skipping backtest.")
        return

    try:
        import joblib
    except ImportError:
        raise ImportError("joblib not installed.")

    model_path = Path(models_dir) / version
    feat_path = Path(features_dir)
    out_path = Path(output_dir)

    # Load models
    long_model_file = model_path / f"{symbol}_long_model.pkl"
    short_model_file = model_path / f"{symbol}_short_model.pkl"
    if not long_model_file.exists() or not short_model_file.exists():
        log.error(f"Models not found in {model_path}. Run step6 first.")
        sys.exit(1)

    long_model = joblib.load(long_model_file)
    short_model = joblib.load(short_model_file)
    log.info(f"[Step7] Loaded models from {model_path}")

    # Load OOS data
    oos_path = model_path / f"{symbol}_oos.parquet"
    if not oos_path.exists():
        log.error(f"OOS data not found: {oos_path}")
        sys.exit(1)

    oos_df = pd.read_parquet(oos_path)
    oos_df["open_time"] = pd.to_datetime(oos_df["open_time"], utc=True)
    oos_df = oos_df.sort_values("open_time").reset_index(drop=True)
    log.info(f"[Step7] OOS bars: {len(oos_df)}")

    META_COLS = {"open_time", "label", "close", "high", "low"}
    feature_cols = [c for c in oos_df.columns if c not in META_COLS]

    backtester = Backtester(
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        leverage=leverage,
        sl_atr_mult=sl_atr_mult,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        max_hold_bars=max_hold_bars,
    )

    result = backtester.simulate(oos_df, long_model, short_model, feature_cols)
    stats = compute_stats(result, initial_capital)

    log.info("[Step7] Backtest results:")
    for k, v in stats.items():
        log.info(f"  {k}: {v}")

    # Save stats
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / f"{version}_{symbol}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save trades
    if len(result["trades"]) > 0:
        result["trades"].to_csv(out_path / f"{version}_{symbol}_trades.csv", index=False)

    build_html_report(
        result=result,
        stats=stats,
        version=version,
        symbol=symbol,
        oos_df=oos_df,
        output_path=out_path / f"{version}_{symbol}_backtest.html",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step7: OOS Backtest")
    parser.add_argument("--version", required=True)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--features_dir", default="data/features")
    parser.add_argument("--output_dir", default="reports")
    parser.add_argument("--label_interval", default="4h")
    parser.add_argument("--initial_capital", type=float, default=100.0)
    parser.add_argument("--risk_per_trade", type=float, default=0.15)
    parser.add_argument("--leverage", type=int, default=5)
    parser.add_argument("--sl_atr_mult", type=float, default=1.5)
    parser.add_argument("--fee_rate", type=float, default=0.0004)
    parser.add_argument("--slippage_rate", type=float, default=0.0002)
    parser.add_argument("--long_threshold", type=float, default=0.55)
    parser.add_argument("--short_threshold", type=float, default=0.55)
    parser.add_argument("--max_hold_bars", type=int, default=48)
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    run(
        version=args.version, symbol=args.symbol,
        models_dir=args.models_dir, features_dir=args.features_dir,
        output_dir=args.output_dir, label_interval=args.label_interval,
        initial_capital=args.initial_capital, risk_per_trade=args.risk_per_trade,
        leverage=args.leverage, sl_atr_mult=args.sl_atr_mult,
        fee_rate=args.fee_rate, slippage_rate=args.slippage_rate,
        long_threshold=args.long_threshold, short_threshold=args.short_threshold,
        max_hold_bars=args.max_hold_bars, skip=args.skip,
    )
