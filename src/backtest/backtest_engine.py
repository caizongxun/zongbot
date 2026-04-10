"""
Step 7 - OOS Backtest Engine
=============================
Rolling bar-by-bar simulation with:
  - Compounding position sizing (15-20% of equity per trade)
  - 5x leverage
  - Strict stop-loss per trade
  - Commission + slippage model
  - Long / Short separate model signals
  - NO lookahead: each bar only sees data up to that point
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    # Capital
    initial_capital: float = 100.0       # USDT
    position_size_pct: float = 0.175     # 17.5% of equity per trade (midpoint 15-20%)
    leverage: float = 5.0

    # Risk management
    stop_loss_pct: float = 0.02          # 2% of entry price (liquidation buffer)
    take_profit_pct: float = 0.0         # 0 = no TP, let model exit
    max_hold_bars: int = 200             # force close after N bars

    # Costs
    commission_pct: float = 0.00045      # 0.045% taker fee (Binance futures)
    slippage_pct: float = 0.0003         # 0.03% slippage estimate

    # Signal thresholds
    long_prob_threshold: float = 0.55    # long model P(positive) >= this -> enter long
    short_prob_threshold: float = 0.55   # short model P(positive) >= this -> enter short
    exit_flat_threshold: float = 0.45    # close position if prob drops below this

    # OOS rolling window
    oos_start: Optional[str] = None
    oos_end: Optional[str] = None

    # Symbol metadata
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"


@dataclass
class Trade:
    entry_time: pd.Timestamp
    direction: str          # 'long' or 'short'
    entry_price: float
    size_usdt: float        # notional after leverage
    stop_price: float
    tp_price: float
    equity_at_entry: float

    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    pnl_usdt: float = 0.0
    pnl_pct: float = 0.0
    commission_usdt: float = 0.0

    @property
    def is_open(self) -> bool:
        return self.exit_time is None


@dataclass
class BacktestResult:
    config: BacktestConfig
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    drawdown_curve: pd.Series = field(default_factory=pd.Series)
    metrics: Dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            f"{'='*50}",
            f"OOS Backtest Results — {self.config.symbol} {self.config.timeframe}",
            f"{'='*50}",
            f"Period     : {self.config.oos_start} -> {self.config.oos_end}",
            f"Init Cap   : ${self.config.initial_capital:.2f}",
            f"Final Cap  : ${m.get('final_equity', 0):.2f}",
            f"Total Ret  : {m.get('total_return_pct', 0):.2f}%",
            f"Max DD     : {m.get('max_drawdown_pct', 0):.2f}%",
            f"Sharpe     : {m.get('sharpe_ratio', 0):.3f}",
            f"Sortino    : {m.get('sortino_ratio', 0):.3f}",
            f"Win Rate   : {m.get('win_rate_pct', 0):.1f}%",
            f"Total Trades: {m.get('total_trades', 0)}",
            f"Profit Factor: {m.get('profit_factor', 0):.3f}",
            f"Avg Trade R : {m.get('avg_r_multiple', 0):.3f}R",
            f"{'='*50}",
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Rolling bar-by-bar OOS simulator.

    Parameters
    ----------
    config : BacktestConfig
    long_probs  : pd.Series  — probability output of long model (index=datetime)
    short_probs : pd.Series  — probability output of short model (index=datetime)
    ohlcv       : pd.DataFrame — OHLCV with DatetimeIndex
    """

    def __init__(
        self,
        config: BacktestConfig,
        long_probs: pd.Series,
        short_probs: pd.Series,
        ohlcv: pd.DataFrame,
    ):
        self.cfg = config
        self.long_probs = long_probs
        self.short_probs = short_probs
        self.ohlcv = ohlcv

    def run(self) -> BacktestResult:
        cfg = self.cfg
        result = BacktestResult(config=cfg)

        # Align data to OOS window
        df = self.ohlcv.copy()
        lp = self.long_probs.reindex(df.index).fillna(0.5)
        sp = self.short_probs.reindex(df.index).fillna(0.5)

        if cfg.oos_start:
            mask = df.index >= pd.Timestamp(cfg.oos_start)
            df = df[mask]
            lp = lp[mask]
            sp = sp[mask]
        if cfg.oos_end:
            mask = df.index <= pd.Timestamp(cfg.oos_end)
            df = df[mask]
            lp = lp[mask]
            sp = sp[mask]

        equity = cfg.initial_capital
        equity_history = []
        current_trade: Optional[Trade] = None
        bars_in_trade = 0

        for i, (ts, row) in enumerate(df.iterrows()):
            bar_open = row["open"]
            bar_high = row["high"]
            bar_low = row["low"]
            bar_close = row["close"]
            long_p = float(lp.loc[ts])
            short_p = float(sp.loc[ts])

            # ---- Manage open trade ----
            if current_trade is not None:
                bars_in_trade += 1
                trade = current_trade

                hit_stop, hit_tp, forced_exit = False, False, False
                exit_price = bar_close
                exit_reason = ""

                if trade.direction == "long":
                    if bar_low <= trade.stop_price:
                        hit_stop = True
                        exit_price = trade.stop_price
                        exit_reason = "stop_loss"
                    elif cfg.take_profit_pct > 0 and bar_high >= trade.tp_price:
                        hit_tp = True
                        exit_price = trade.tp_price
                        exit_reason = "take_profit"
                    elif long_p < cfg.exit_flat_threshold:
                        exit_price = bar_close
                        exit_reason = "signal_exit"
                        forced_exit = True
                elif trade.direction == "short":
                    if bar_high >= trade.stop_price:
                        hit_stop = True
                        exit_price = trade.stop_price
                        exit_reason = "stop_loss"
                    elif cfg.take_profit_pct > 0 and bar_low <= trade.tp_price:
                        hit_tp = True
                        exit_price = trade.tp_price
                        exit_reason = "take_profit"
                    elif short_p < cfg.exit_flat_threshold:
                        exit_price = bar_close
                        exit_reason = "signal_exit"
                        forced_exit = True

                if bars_in_trade >= cfg.max_hold_bars:
                    forced_exit = True
                    exit_price = bar_close
                    exit_reason = "max_hold"

                if hit_stop or hit_tp or forced_exit:
                    # Apply slippage on exit
                    slip = exit_price * cfg.slippage_pct
                    if trade.direction == "long":
                        exit_price -= slip
                    else:
                        exit_price += slip

                    # Compute PnL
                    notional = trade.size_usdt
                    if trade.direction == "long":
                        raw_pnl = (exit_price - trade.entry_price) / trade.entry_price * notional
                    else:
                        raw_pnl = (trade.entry_price - exit_price) / trade.entry_price * notional

                    commission = (notional * cfg.commission_pct) + (notional * cfg.commission_pct)
                    net_pnl = raw_pnl - commission

                    trade.exit_time = ts
                    trade.exit_price = exit_price
                    trade.exit_reason = exit_reason
                    trade.pnl_usdt = net_pnl
                    trade.pnl_pct = net_pnl / trade.equity_at_entry * 100
                    trade.commission_usdt = commission

                    equity += net_pnl
                    equity = max(equity, 0.01)  # prevent negative equity
                    result.trades.append(trade)
                    current_trade = None
                    bars_in_trade = 0

            # ---- Consider new trade (only if flat) ----
            if current_trade is None:
                direction = None
                if long_p >= cfg.long_prob_threshold and long_p > short_p:
                    direction = "long"
                elif short_p >= cfg.short_prob_threshold and short_p > long_p:
                    direction = "short"

                if direction is not None:
                    # Entry price with slippage
                    entry_price = bar_close
                    slip = entry_price * cfg.slippage_pct
                    if direction == "long":
                        entry_price += slip
                        stop_price = entry_price * (1 - cfg.stop_loss_pct)
                        tp_price = entry_price * (1 + cfg.take_profit_pct) if cfg.take_profit_pct > 0 else float("inf")
                    else:
                        entry_price -= slip
                        stop_price = entry_price * (1 + cfg.stop_loss_pct)
                        tp_price = entry_price * (1 - cfg.take_profit_pct) if cfg.take_profit_pct > 0 else 0.0

                    # Position sizing: % of equity * leverage = notional
                    risk_equity = equity * cfg.position_size_pct
                    notional = risk_equity * cfg.leverage
                    commission = notional * cfg.commission_pct
                    equity -= commission  # deduct entry commission immediately

                    current_trade = Trade(
                        entry_time=ts,
                        direction=direction,
                        entry_price=entry_price,
                        size_usdt=notional,
                        stop_price=stop_price,
                        tp_price=tp_price,
                        equity_at_entry=equity,
                    )
                    bars_in_trade = 0

            equity_history.append((ts, equity))

        # Close any open trade at end
        if current_trade is not None:
            last_row = df.iloc[-1]
            exit_price = last_row["close"]
            trade = current_trade
            notional = trade.size_usdt
            if trade.direction == "long":
                raw_pnl = (exit_price - trade.entry_price) / trade.entry_price * notional
            else:
                raw_pnl = (trade.entry_price - exit_price) / trade.entry_price * notional
            commission = notional * cfg.commission_pct
            net_pnl = raw_pnl - commission
            trade.exit_time = df.index[-1]
            trade.exit_price = exit_price
            trade.exit_reason = "end_of_data"
            trade.pnl_usdt = net_pnl
            trade.pnl_pct = net_pnl / trade.equity_at_entry * 100
            trade.commission_usdt = commission
            equity += net_pnl
            result.trades.append(trade)

        # Build equity curve
        eq_series = pd.Series(
            {t: e for t, e in equity_history}, name="equity"
        )
        result.equity_curve = eq_series

        # Drawdown
        rolling_max = eq_series.cummax()
        dd = (eq_series - rolling_max) / rolling_max * 100
        result.drawdown_curve = dd

        # Compute metrics
        result.metrics = self._compute_metrics(result, cfg)
        return result

    @staticmethod
    def _compute_metrics(result: BacktestResult, cfg: BacktestConfig) -> Dict:
        trades = result.trades
        eq = result.equity_curve

        if len(eq) == 0:
            return {}

        final_equity = float(eq.iloc[-1])
        total_return_pct = (final_equity - cfg.initial_capital) / cfg.initial_capital * 100
        max_dd = float(result.drawdown_curve.min())

        pnls = [t.pnl_usdt for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / max(len(pnls), 1) * 100
        profit_factor = abs(sum(wins)) / max(abs(sum(losses)), 1e-9)

        # Daily returns for Sharpe/Sortino
        daily_eq = eq.resample("1D").last().ffill()
        daily_ret = daily_eq.pct_change().dropna()
        rf_daily = 0.0
        excess = daily_ret - rf_daily
        sharpe = float(excess.mean() / excess.std() * np.sqrt(252)) if excess.std() > 0 else 0.0
        downside = daily_ret[daily_ret < 0].std()
        sortino = float(excess.mean() / downside * np.sqrt(252)) if downside > 0 else 0.0

        # Average R multiple (using stop distance as 1R)
        r_multiples = []
        for t in trades:
            if t.entry_price and t.stop_price:
                r = abs(t.entry_price - t.stop_price)
                if r > 0:
                    r_multiples.append(t.pnl_usdt / (t.size_usdt * (r / t.entry_price)))
        avg_r = float(np.mean(r_multiples)) if r_multiples else 0.0

        long_trades = [t for t in trades if t.direction == "long"]
        short_trades = [t for t in trades if t.direction == "short"]

        return {
            "final_equity": final_equity,
            "total_return_pct": total_return_pct,
            "max_drawdown_pct": max_dd,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "win_rate_pct": win_rate,
            "total_trades": len(trades),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "profit_factor": profit_factor,
            "avg_r_multiple": avg_r,
            "total_commission": sum(t.commission_usdt for t in trades),
            "best_trade_pct": max([t.pnl_pct for t in trades], default=0),
            "worst_trade_pct": min([t.pnl_pct for t in trades], default=0),
            "avg_bars_held": np.mean([
                (t.exit_time - t.entry_time).total_seconds() / 3600
                for t in trades if t.exit_time
            ]) if trades else 0,
        }
