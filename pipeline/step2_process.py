"""Step 2: Decompress, merge, and preprocess raw Binance Vision data.

Features:
- Auto-detects available RAM and uses chunk-based processing to avoid OOM.
- Merges multi-timeframe klines, aggTrades, fundingRate into Parquet files.
- Resamples aggTrades to kline timeframes for volume/trade metrics.

Usage:
    python pipeline/step2_process.py \
        --symbol BTCUSDT \
        --start_date 2023-01-01 \
        --end_date 2024-12-31 \
        --raw_dir data/raw \
        --processed_dir data/processed \
        --intervals 5m 15m 1h 4h 1d \
        --ram_fraction 0.4
"""

import os
import sys
import gc
import zipfile
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import psutil
import pandas as pd
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore",
]
AGGTRADE_COLS = [
    "agg_id", "price", "quantity", "first_trade_id", "last_trade_id",
    "timestamp", "is_buyer_maker", "is_best_match",
]
FUNDING_COLS = ["symbol", "fundingTime", "fundingRate", "markPrice"]


def available_ram_bytes(fraction: float = 0.4) -> int:
    mem = psutil.virtual_memory()
    return int(mem.available * fraction)


def estimate_chunk_size(file_size_bytes: int, ram_budget: int, row_bytes: int = 200) -> int:
    """Estimate safe chunksize given RAM budget."""
    max_rows = ram_budget // row_bytes
    approx_rows = file_size_bytes // 50  # rough compressed ratio
    if approx_rows <= max_rows:
        return 0  # read all at once
    return max(10000, max_rows // 4)


def read_zip_csv(zip_path: Path, columns: List[str], chunksize: Optional[int] = None) -> pd.DataFrame:
    """Read CSV inside a ZIP with optional chunking."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        csv_name = next((n for n in names if n.endswith(".csv")), names[0])
        with zf.open(csv_name) as f:
            data = f.read()
    bio = BytesIO(data)
    if chunksize and chunksize > 0:
        chunks = []
        for chunk in pd.read_csv(bio, header=None, names=columns, chunksize=chunksize):
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(bio, header=None, names=columns)
    return df


def process_klines(symbol: str, interval: str, start_date: str, end_date: str,
                   raw_dir: Path, out_dir: Path, ram_fraction: float) -> Optional[Path]:
    """Process all daily kline ZIPs for a given interval into one Parquet."""
    src_dir = raw_dir / "klines" / symbol / interval
    if not src_dir.exists():
        log.warning(f"[Step2] klines dir not found: {src_dir}")
        return None

    zip_files = sorted(src_dir.glob("*.zip"))
    if not zip_files:
        log.warning(f"[Step2] No klines ZIPs found in {src_dir}")
        return None

    ram = available_ram_bytes(ram_fraction)
    all_frames = []
    for zp in tqdm(zip_files, desc=f"klines/{interval}"):
        cs = estimate_chunk_size(zp.stat().st_size, ram)
        try:
            df = read_zip_csv(zp, KLINE_COLS, chunksize=cs)
            all_frames.append(df)
        except Exception as e:
            log.warning(f"Failed to read {zp}: {e}")
        if len(all_frames) % 100 == 0:
            gc.collect()

    if not all_frames:
        return None

    df = pd.concat(all_frames, ignore_index=True)
    del all_frames
    gc.collect()

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_volume", "taker_buy_quote_volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    df = df[(df["open_time"] >= start_date) & (df["open_time"] <= end_date)]

    out_path = out_dir / "klines" / symbol / f"{symbol}_{interval}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"[Step2] Saved klines/{interval}: {len(df)} rows -> {out_path}")
    del df
    gc.collect()
    return out_path


def process_aggtrades(symbol: str, start_date: str, end_date: str,
                      raw_dir: Path, out_dir: Path, ram_fraction: float) -> Optional[Path]:
    """Process aggTrades ZIPs and resample to 1-minute aggregates."""
    src_dir = raw_dir / "aggtrades" / symbol
    if not src_dir.exists():
        log.warning(f"[Step2] aggtrades dir not found: {src_dir}")
        return None

    zip_files = sorted(src_dir.glob("*.zip"))
    if not zip_files:
        return None

    ram = available_ram_bytes(ram_fraction)
    all_frames = []
    for zp in tqdm(zip_files, desc="aggTrades"):
        cs = estimate_chunk_size(zp.stat().st_size, ram, row_bytes=80)
        try:
            df = read_zip_csv(zp, AGGTRADE_COLS, chunksize=cs)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)
            # Resample to 1m buckets immediately to save RAM
            df = df.set_index("timestamp")
            resampled = df.resample("1min").agg(
                agg_trades_count=("agg_id", "count"),
                agg_buy_volume=("quantity", lambda x: x[~df.loc[x.index, "is_buyer_maker"]].sum() if len(x) > 0 else 0),
                agg_sell_volume=("quantity", lambda x: x[df.loc[x.index, "is_buyer_maker"]].sum() if len(x) > 0 else 0),
                agg_total_volume=("quantity", "sum"),
                agg_vwap=("price", lambda x: np.average(x, weights=df.loc[x.index, "quantity"]) if len(x) > 0 else np.nan),
                agg_price_std=("price", "std"),
            ).reset_index()
            all_frames.append(resampled)
        except Exception as e:
            log.warning(f"Failed to process aggTrades {zp}: {e}")
        gc.collect()

    if not all_frames:
        return None

    df = pd.concat(all_frames, ignore_index=True)
    del all_frames
    gc.collect()

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)]

    out_path = out_dir / "aggtrades" / symbol / f"{symbol}_aggtrades_1m.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"[Step2] Saved aggTrades: {len(df)} rows -> {out_path}")
    del df
    gc.collect()
    return out_path


def process_funding(symbol: str, start_date: str, end_date: str,
                    raw_dir: Path, out_dir: Path, ram_fraction: float) -> Optional[Path]:
    """Process funding rate ZIPs into single Parquet."""
    src_dir = raw_dir / "fundingrate" / symbol
    if not src_dir.exists():
        log.warning(f"[Step2] fundingrate dir not found: {src_dir}")
        return None

    zip_files = sorted(src_dir.glob("*.zip"))
    if not zip_files:
        return None

    all_frames = []
    for zp in tqdm(zip_files, desc="fundingRate"):
        try:
            df = read_zip_csv(zp, FUNDING_COLS)
            all_frames.append(df)
        except Exception as e:
            log.warning(f"Failed to read {zp}: {e}")

    if not all_frames:
        return None

    df = pd.concat(all_frames, ignore_index=True)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
    df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
    df = df.sort_values("fundingTime").reset_index(drop=True)
    df = df[(df["fundingTime"] >= start_date) & (df["fundingTime"] <= end_date)]

    out_path = out_dir / "fundingrate" / symbol / f"{symbol}_fundingrate.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    log.info(f"[Step2] Saved fundingRate: {len(df)} rows -> {out_path}")
    return out_path


def run(
    symbol: str,
    start_date: str,
    end_date: str,
    raw_dir: str,
    processed_dir: str,
    intervals: List[str],
    ram_fraction: float = 0.4,
    skip: bool = False,
):
    if skip:
        log.info("[Step2] skip=True, skipping processing.")
        return

    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)

    log.info(f"[Step2] Processing {symbol} from {start_date} to {end_date}")

    for interval in intervals:
        process_klines(symbol, interval, start_date, end_date, raw_path, proc_path, ram_fraction)

    process_aggtrades(symbol, start_date, end_date, raw_path, proc_path, ram_fraction)
    process_funding(symbol, start_date, end_date, raw_path, proc_path, ram_fraction)

    log.info("[Step2] Processing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step2: Process raw Binance Vision data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "1h", "4h", "1d"])
    parser.add_argument("--ram_fraction", type=float, default=0.4,
                        help="Fraction of available RAM to use for chunked processing")
    parser.add_argument("--skip", action="store_true")
    args = parser.parse_args()
    run(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        intervals=args.intervals,
        ram_fraction=args.ram_fraction,
        skip=args.skip,
    )
