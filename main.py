#!/usr/bin/env python3
"""ZongBot Main Entry Point

This script demonstrates the workflow for Phase 1: Data Collection and Processing
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.binance_fetcher import BinanceFetcher
from src.data.data_processor import DataProcessor
from src.utils.logger import get_logger
from src.utils.config import load_symbols, load_indicators
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = get_logger(__name__)


def main():
    """Main execution function for Phase 1: Data Pipeline"""
    
    logger.info("="*50)
    logger.info("ZongBot Phase 1: Data Collection Pipeline")
    logger.info("="*50)
    
    # Step 1: Initialize Binance Fetcher
    logger.info("\n[Step 1] Initializing Binance API Fetcher...")
    try:
        fetcher = BinanceFetcher(testnet=False)
        logger.info("✓ Binance Fetcher initialized successfully")
    except ValueError as e:
        logger.error(f"✗ Failed to initialize: {e}")
        logger.error("Please check your BINANCE_API_KEY and BINANCE_API_SECRET in .env")
        return
    
    # Step 2: Load Configuration
    logger.info("\n[Step 2] Loading configuration...")
    try:
        symbols = load_symbols("config/symbols.json")
        indicators = load_indicators("config/indicators.json")
        logger.info(f"✓ Loaded {len(symbols)} symbols and {indicators['total_indicators']} indicators")
        logger.info(f"  Symbols: {', '.join(symbols[:5])}...")
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return
    
    # Step 3: Fetch Data
    logger.info("\n[Step 3] Fetching cryptocurrency data...")
    logger.info(f"  This will fetch {len(symbols)} symbols across 3 timeframes (15m, 1h, 4h)")
    logger.info(f"  Estimated API calls: {len(symbols) * 3}")
    logger.info("  Note: Starting with a small sample for demonstration")
    
    try:
        # For demonstration, fetch only first 3 symbols with limit=10
        test_symbols = symbols[:3]
        test_limit = 10
        
        logger.info(f"\n  Demo: Fetching {test_symbols} with limit={test_limit} candles...")
        fetched_data = fetcher.fetch_multiple_symbols(
            test_symbols,
            timeframe="1h",
            limit=test_limit,
            max_workers=2
        )
        
        logger.info(f"✓ Successfully fetched {len(fetched_data)} symbols")
        
        # Step 4: Process Data
        logger.info("\n[Step 4] Processing and cleaning data...")
        for symbol, df in fetched_data.items():
            cleaned_df = DataProcessor.clean_ohlcv(df)
            logger.info(f"  {symbol}: {len(df)} → {len(cleaned_df)} candles (cleaned)")
        
        logger.info("✓ Data processing completed")
        
        # Step 5: Summary
        logger.info("\n" + "="*50)
        logger.info("Phase 1 Summary")
        logger.info("="*50)
        logger.info(f"Total symbols configured: {len(symbols)}")
        logger.info(f"Timeframes: 15m, 1h, 4h")
        logger.info(f"Demo fetched: {len(fetched_data)} symbols with {test_limit} candles each")
        logger.info("\nNext Steps:")
        logger.info("1. Set up HuggingFace repositories for data and models")
        logger.info("2. Configure your Binance API credentials fully")
        logger.info("3. Run: python main.py --mode full --symbols all")
        logger.info("4. Proceed to Phase 2: Feature Engineering")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"✗ Error during data fetching: {e}")
        logger.error("Please check your network connection and API credentials")
        return


if __name__ == "__main__":
    main()
