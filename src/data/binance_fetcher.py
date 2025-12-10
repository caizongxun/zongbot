"""Binance API Data Fetcher

This module handles fetching cryptocurrency data from Binance API.
Supports multiple timeframes (15m, 1h, 4h) and multiple symbols.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import pandas as pd
import numpy as np
from binance.spot import Spot
from binance.error import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BinanceFetcher:
    """Binance API wrapper for fetching OHLCV data."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
    ):
        """Initialize Binance API client.
        
        Args:
            api_key: Binance API key. Defaults to env var BINANCE_API_KEY
            api_secret: Binance API secret. Defaults to env var BINANCE_API_SECRET
            testnet: Whether to use testnet. Defaults to False
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET")
        self.testnet = testnet
        
        if not self.api_key or not self.api_secret:
            raise ValueError(
                "Binance API credentials not found. "
                "Please set BINANCE_API_KEY and BINANCE_API_SECRET environment variables."
            )
        
        # Initialize Binance client
        self.client = Spot(
            api_key=self.api_key,
            api_secret=self.api_secret,
            base_url="https://testnet.binance.vision" if testnet else "https://api.binance.com"
        )
        
        # Timeframe mapping (Binance uses string intervals)
        self.timeframe_map = {
            "15m": "15m",
            "1h": "1h",
            "4h": "4h",
        }
        
        logger.info(f"Binance Fetcher initialized. Testnet: {testnet}")
    
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 1000,
        start_time: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Binance.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            timeframe: Timeframe ('15m', '1h', '4h')
            limit: Number of candles to fetch (max 1000)
            start_time: Start time in milliseconds (optional)
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if timeframe not in self.timeframe_map:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        if limit > 1000:
            logger.warning(f"Limit {limit} exceeds max 1000. Setting to 1000.")
            limit = 1000
        
        try:
            klines = self.client.klines(
                symbol=symbol,
                interval=self.timeframe_map[timeframe],
                startTime=start_time,
                limit=limit
            )
            
            # Parse kline data
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # Convert types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
            
            # Select relevant columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Reset index
            df = df.reset_index(drop=True)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df
        
        except ClientError as e:
            logger.error(f"Binance API error for {symbol}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching {symbol}: {e}")
            raise
    
    def fetch_multiple_symbols(
        self,
        symbols: List[str],
        timeframe: str = "1h",
        limit: int = 1000,
        max_workers: int = 5,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols concurrently.
        
        Args:
            symbols: List of trading pairs
            timeframe: Timeframe for all symbols
            limit: Number of candles per symbol
            max_workers: Number of concurrent workers
        
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        results = {}
        errors = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(self.fetch_ohlcv, symbol, timeframe, limit): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error fetching {symbol}: {e}")
                    errors[symbol] = str(e)
        
        if errors:
            logger.warning(f"Failed to fetch {len(errors)} symbols: {list(errors.keys())}")
        
        logger.info(f"Successfully fetched {len(results)} symbols")
        return results
    
    def fetch_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str] = None,
        limit: int = 1000,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple timeframes of a single symbol.
        
        Args:
            symbol: Trading pair
            timeframes: List of timeframes. Defaults to ['15m', '1h', '4h']
            limit: Number of candles per timeframe
        
        Returns:
            Dictionary mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = ["15m", "1h", "4h"]
        
        results = {}
        for timeframe in timeframes:
            try:
                results[timeframe] = self.fetch_ohlcv(symbol, timeframe, limit)
            except Exception as e:
                logger.error(f"Error fetching {symbol} {timeframe}: {e}")
        
        return results
    
    def fetch_all_data(
        self,
        symbols_file: str = "config/symbols.json",
        limit: int = 1000,
        max_workers: int = 5,
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Fetch data for all symbols and timeframes.
        
        Args:
            symbols_file: Path to JSON file containing symbols list
            limit: Number of candles per symbol-timeframe
            max_workers: Number of concurrent workers
        
        Returns:
            Nested dictionary: {symbol: {timeframe: DataFrame}}
        """
        # Load symbols
        with open(symbols_file, 'r') as f:
            config = json.load(f)
        
        symbols = config.get('symbols', [])
        timeframes = config.get('timeframes', ['15m', '1h', '4h'])
        
        logger.info(f"Fetching data for {len(symbols)} symbols across {len(timeframes)} timeframes")
        
        all_data = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {
                executor.submit(
                    self.fetch_multiple_timeframes,
                    symbol,
                    timeframes,
                    limit
                ): symbol
                for symbol in symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    all_data[symbol] = future.result()
                    logger.info(f"✓ Completed {symbol}")
                except Exception as e:
                    logger.error(f"✗ Failed {symbol}: {e}")
        
        logger.info(f"Fetched data for {len(all_data)} symbols")
        return all_data


def main():
    """Example usage of BinanceFetcher."""
    
    # Initialize fetcher
    fetcher = BinanceFetcher(testnet=False)
    
    # Example 1: Fetch single symbol
    print("\n=== Fetching BTC/USDT (1h) ===")
    btc_df = fetcher.fetch_ohlcv("BTCUSDT", timeframe="1h", limit=10)
    print(btc_df.head())
    print(f"Shape: {btc_df.shape}")
    
    # Example 2: Fetch multiple symbols
    print("\n=== Fetching multiple symbols ===")
    symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    multi_df = fetcher.fetch_multiple_symbols(symbols, timeframe="1h", limit=5)
    for symbol, df in multi_df.items():
        print(f"{symbol}: {len(df)} candles")
    
    # Example 3: Fetch multiple timeframes
    print("\n=== Fetching multiple timeframes ===")
    timeframe_data = fetcher.fetch_multiple_timeframes("ETHUSDT", limit=5)
    for tf, df in timeframe_data.items():
        print(f"{tf}: {len(df)} candles")


if __name__ == "__main__":
    main()
