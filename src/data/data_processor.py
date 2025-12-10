"""Data Processing and Cleaning Module

Handles data cleaning, validation, and transformation.
"""

import logging
from typing import Dict, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class DataProcessor:
    """Process and clean cryptocurrency OHLCV data."""
    
    @staticmethod
    def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Remove rows with NaN values in critical columns
        critical_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df = df.dropna(subset=critical_cols)
        
        # Validate OHLC relationships
        df = df[(df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])]
        
        # Remove zero/negative volumes
        df = df[df['volume'] > 0]
        
        # Remove zero prices
        df = df[(df['open'] > 0) & (df['close'] > 0)]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    @staticmethod
    def handle_missing_data(
        df: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """Handle missing data in time series.
        
        Args:
            df: DataFrame with time series data
            method: 'forward_fill', 'interpolate', or 'drop'
        
        Returns:
            DataFrame with missing data handled
        """
        df = df.copy()
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
        
        return df
    
    @staticmethod
    def normalize_prices(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize price columns to [0, 1] range.
        
        Args:
            df: DataFrame with price data
        
        Returns:
            DataFrame with normalized prices
        """
        df = df.copy()
        price_cols = ['open', 'high', 'low', 'close']
        
        for col in price_cols:
            if col in df.columns:
                min_val = df[col].min()
                max_val = df[col].max()
                df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
        
        return df
    
    @staticmethod
    def calculate_returns(df: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """Calculate returns from price data.
        
        Args:
            df: DataFrame with price data
            column: Column name to calculate returns from
        
        Returns:
            DataFrame with added returns column
        """
        df = df.copy()
        df['returns'] = df[column].pct_change()
        df['log_returns'] = np.log(df[column] / df[column].shift(1))
        return df
    
    @staticmethod
    def detect_outliers(
        df: pd.DataFrame,
        column: str = 'close',
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """Detect outliers in data.
        
        Args:
            df: DataFrame with data
            column: Column to check for outliers
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outlier flag
        """
        df = df.copy()
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df['is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df['is_outlier'] = z_scores > threshold
        
        return df
    
    @staticmethod
    def aggregate_timeframe(
        df: pd.DataFrame,
        from_tf: str = "1h",
        to_tf: str = "4h"
    ) -> pd.DataFrame:
        """Aggregate data from smaller to larger timeframe.
        
        Args:
            df: DataFrame with time series data
            from_tf: Source timeframe
            to_tf: Target timeframe
        
        Returns:
            Aggregated DataFrame
        """
        df = df.copy()
        df = df.set_index('timestamp')
        
        agg_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        df_agg = df.resample(to_tf).agg(agg_rules)
        df_agg = df_agg.dropna()
        df_agg = df_agg.reset_index()
        
        logger.info(f"Aggregated {len(df)} rows to {len(df_agg)} rows ({from_tf} -> {to_tf})")
        return df_agg
    
    @staticmethod
    def prepare_for_training(
        df: pd.DataFrame,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1
    ) -> tuple:
        """Split data for training, validation, and testing.
        
        Args:
            df: DataFrame with all data
            test_ratio: Ratio of test data
            val_ratio: Ratio of validation data
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        n = len(df)
        test_size = int(n * test_ratio)
        val_size = int(n * val_ratio)
        train_size = n - test_size - val_size
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        logger.info(f"Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
        return train_df, val_df, test_df
