"""Configuration management utilities."""

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_file: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_file: Path to configuration file
    
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def load_symbols(symbols_file: str = "config/symbols.json") -> list:
    """Load cryptocurrency symbols from JSON file.
    
    Args:
        symbols_file: Path to symbols JSON file
    
    Returns:
        List of symbols
    """
    with open(symbols_file, 'r') as f:
        data = json.load(f)
    
    return data.get('symbols', [])


def load_indicators(indicators_file: str = "config/indicators.json") -> Dict:
    """Load technical indicators configuration.
    
    Args:
        indicators_file: Path to indicators JSON file
    
    Returns:
        Indicators configuration dictionary
    """
    with open(indicators_file, 'r') as f:
        data = json.load(f)
    
    return data
