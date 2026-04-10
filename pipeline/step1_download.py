"""Step 1: Download raw data from Binance Vision.

Usage:
    python pipeline/step1_download.py \
        --symbol BTCUSDT \
        --start_date 2023-01-01 \
        --end_date 2024-12-31 \
        --data_dir data/raw \
        --intervals 5m 15m 1h 4h 1d \
        --data_types klines aggtrades fundingrate \
        --workers 4
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily"


def build_url(data_type: str, symbol: str, interval: Optional[str], year: int, month: int, day: int) -> str:
    """Build Binance Vision download URL."""
    date_str = f"{year:04d}-{month:02d}-{day:02d}"
    if data_type == "klines":
        return f"{BASE_URL}/klines/{symbol}/{interval}/{symbol}-{interval}-{date_str}.zip"
    elif data_type == "aggtrades":
        return f"{BASE_URL}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
    elif data_type == "fundingrate":
        return f"{BASE_URL}/fundingRate/{symbol}/{symbol}-fundingRate-{date_str}.zip"
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a single file with retry logic."""
    if dest.exists():
        log.debug(f"Already exists: {dest}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 404:
                log.debug(f"Not found (404): {url}")
                return False
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
            return True
        except Exception as e:
            log.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    return False


def date_range(start: date, end: date) -> List[date]:
    """Generate list of dates inclusive."""
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def build_task_list(
    symbol: str,
    start: date,
    end: date,
    data_dir: Path,
    intervals: List[str],
    data_types: List[str],
) -> List[tuple]:
    """Build list of (url, dest_path) tuples."""
    tasks = []
    for d in date_range(start, end):
        for dtype in data_types:
            if dtype == "klines":
                for interval in intervals:
                    url = build_url(dtype, symbol, interval, d.year, d.month, d.day)
                    dest = data_dir / dtype / symbol / interval / f"{symbol}-{interval}-{d}.zip"
                    tasks.append((url, dest))
            elif dtype == "aggtrades":
                url = build_url(dtype, symbol, None, d.year, d.month, d.day)
                dest = data_dir / dtype / symbol / f"{symbol}-aggTrades-{d}.zip"
                tasks.append((url, dest))
            elif dtype == "fundingrate":
                url = build_url(dtype, symbol, None, d.year, d.month, d.day)
                dest = data_dir / dtype / symbol / f"{symbol}-fundingRate-{d}.zip"
                tasks.append((url, dest))
    return tasks


def run(
    symbol: str,
    start_date: str,
    end_date: str,
    data_dir: str,
    intervals: List[str],
    data_types: List[str],
    workers: int = 4,
    skip: bool = False,
):
    if skip:
        log.info("[Step1] skip=True, skipping download.")
        return

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    data_path = Path(data_dir)

    log.info(f"[Step1] Downloading {symbol} from {start} to {end}")
    log.info(f"[Step1] data_types={data_types}, intervals={intervals}")

    tasks = build_task_list(symbol, start, end, data_path, intervals, data_types)
    log.info(f"[Step1] Total tasks: {len(tasks)}")

    success, fail = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_file, url, dest): (url, dest) for url, dest in tasks}
        with tqdm(total=len(futures), desc="Downloading", unit="file") as pbar:
            for future in as_completed(futures):
                ok = future.result()
                if ok:
                    success += 1
                else:
                    fail += 1
                pbar.update(1)

    log.info(f"[Step1] Done. success={success}, skipped/missing={fail}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step1: Download Binance Vision data")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--start_date", required=True)
    parser.add_argument("--end_date", required=True)
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--intervals", nargs="+", default=["5m", "15m", "1h", "4h", "1d"])
    parser.add_argument("--data_types", nargs="+", default=["klines", "aggtrades", "fundingrate"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip", action="store_true", help="Skip this step")
    args = parser.parse_args()
    run(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        data_dir=args.data_dir,
        intervals=args.intervals,
        data_types=args.data_types,
        workers=args.workers,
        skip=args.skip,
    )
