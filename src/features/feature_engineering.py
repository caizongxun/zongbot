"""Feature Engineering Module - Phase 2

Implements 15+ technical indicators and volatility calculations for model training.
"""

import logging
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Calculate technical indicators and features for time series."""
    
    # Moving Averages
    @staticmethod
    def calculate_sma(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
        """Calculate Simple Moving Average.
        
        Args:
            df: DataFrame with 'close' column
            periods: List of periods to calculate
        
        Returns:
            DataFrame with SMA columns added
        """
        if periods is None:
            periods = [20, 50, 200]
        
        df = df.copy()
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, periods: list = None) -> pd.DataFrame:
        """Calculate Exponential Moving Average.
        
        Args:
            df: DataFrame with 'close' column
            periods: List of periods to calculate
        
        Returns:
            DataFrame with EMA columns added
        """
        if periods is None:
            periods = [12, 26]
        
        df = df.copy()
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        return df
    
    # Momentum Indicators
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with 'close' column
            period: RSI period
        
        Returns:
            DataFrame with RSI column added
        """
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame with 'close' column
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        
        Returns:
            DataFrame with MACD columns added
        """
        df = df.copy()
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        return df
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
        """Calculate Stochastic Oscillator.
        
        Args:
            df: DataFrame with OHLC data
            k_period: K period
            d_period: D period
        
        Returns:
            DataFrame with %K and %D columns added
        """
        df = df.copy()
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        
        return df
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index.
        
        Args:
            df: DataFrame with OHLC data
            period: CCI period
        
        Returns:
            DataFrame with CCI column added
        """
        df = df.copy()
        tp = (df['high'] + df['low'] + df['close']) / 3  # Typical Price
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        return df
    
    # Volatility Indicators
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with 'close' column
            period: MA period
            std_dev: Standard deviation multiplier
        
        Returns:
            DataFrame with BB columns added
        """
        df = df.copy()
        sma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df['bb_upper'] = sma + (std * std_dev)
        df['bb_middle'] = sma
        df['bb_lower'] = sma - (std * std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR period
        
        Returns:
            DataFrame with ATR column added
        """
        df = df.copy()
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def calculate_historical_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Historical Volatility (Standard Deviation of Returns).
        
        Args:
            df: DataFrame with 'close' column
            period: Rolling window period
        
        Returns:
            DataFrame with historical volatility column added
        """
        df = df.copy()
        returns = df['close'].pct_change()
        df['volatility_hist'] = returns.rolling(window=period).std()
        
        return df
    
    @staticmethod
    def calculate_parkinson_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Parkinson Volatility (uses high-low range).
        
        Args:
            df: DataFrame with OHLC data
            period: Rolling window period
        
        Returns:
            DataFrame with Parkinson volatility column added
        """
        df = df.copy()
        hl_ratio = df['high'] / df['low']
        parkinson = np.sqrt(np.log(hl_ratio) ** 2 / (4 * np.log(2)))
        df['volatility_parkinson'] = parkinson.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def calculate_garman_klass_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Garman-Klass Volatility.
        
        Args:
            df: DataFrame with OHLC data
            period: Rolling window period
        
        Returns:
            DataFrame with GK volatility column added
        """
        df = df.copy()
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        gk = 0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2
        df['volatility_gk'] = np.sqrt(gk.rolling(window=period).mean())
        
        return df
    
    # Volume Indicators
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume.
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            DataFrame with OBV column added
        """
        df = df.copy()
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv
        
        return df
    
    # Trend Indicators
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index.
        
        Args:
            df: DataFrame with OHLC data
            period: ADX period
        
        Returns:
            DataFrame with ADX column added
        """
        df = df.copy()
        
        # Calculate directional movements
        up_move = df['high'].diff()
        down_move = -df['low'].diff()
        
        pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([pd.Series(high_low), pd.Series(high_close), pd.Series(low_close)], axis=1).max(axis=1)
        
        # Directional Indicators
        pos_di = 100 * pos_dm.rolling(window=period).mean() / tr.rolling(window=period).mean()
        neg_di = 100 * neg_dm.rolling(window=period).mean() / tr.rolling(window=period).mean()
        
        # ADX
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        df['adx'] = dx.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame, include_volatility: bool = True) -> pd.DataFrame:
        """Calculate all technical indicators at once.
        
        Args:
            df: DataFrame with OHLCV data
            include_volatility: Whether to include volatility calculations
        
        Returns:
            DataFrame with all features added
        """
        df = df.copy()
        
        logger.info("Calculating technical indicators...")
        
        # Moving Averages
        df = FeatureEngineer.calculate_sma(df)
        df = FeatureEngineer.calculate_ema(df)
        
        # Momentum
        df = FeatureEngineer.calculate_rsi(df)
        df = FeatureEngineer.calculate_macd(df)
        df = FeatureEngineer.calculate_stochastic(df)
        df = FeatureEngineer.calculate_cci(df)
        
        # Volatility
        df = FeatureEngineer.calculate_bollinger_bands(df)
        df = FeatureEngineer.calculate_atr(df)
        
        if include_volatility:
            logger.info("Calculating volatility metrics...")
            df = FeatureEngineer.calculate_historical_volatility(df)
            df = FeatureEngineer.calculate_parkinson_volatility(df)
            df = FeatureEngineer.calculate_garman_klass_volatility(df)
        
        # Volume
        df = FeatureEngineer.calculate_obv(df)
        
        # Trend
        df = FeatureEngineer.calculate_adx(df)
        
        logger.info(f"Feature calculation complete. Total columns: {len(df.columns)}")
        return df


def main():
    """Example usage of FeatureEngineer."""
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='1h')
    data = {
        'timestamp': dates,
        'open': np.random.uniform(40000, 50000, 200),
        'high': np.random.uniform(40000, 50000, 200),
        'low': np.random.uniform(40000, 50000, 200),
        'close': np.random.uniform(40000, 50000, 200),
        'volume': np.random.uniform(100, 1000, 200)
    }
    df = pd.DataFrame(data)
    
    # Ensure high >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    print("Original data shape:", df.shape)
    print("\nCalculating features...")
    
    df_features = FeatureEngineer.calculate_all_features(df)
    
    print(f"Features calculated shape: {df_features.shape}")
    print(f"\nNew columns: {[col for col in df_features.columns if col not in df.columns]}")
    print(f"\nSample data with features:")
    print(df_features[['timestamp', 'close', 'rsi', 'volatility_hist', 'atr', 'adx']].tail(10))


if __name__ == "__main__":
    main()
