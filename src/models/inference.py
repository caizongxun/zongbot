"""Model Inference Module - Phase 3-4

Run inference on trained models to generate trading signals.
"""

import logging
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Predictor:
    """Run inference on trained models."""
    
    def __init__(
        self,
        model,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        model_path: Optional[str] = None,
    ):
        """Initialize predictor.
        
        Args:
            model: Trained model
            device: Device to use
            model_path: Path to load saved model weights
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()
        
        if model_path:
            self.load_model(model_path)
        
        logger.info(f"Predictor initialized on device: {device}")
    
    def load_model(self, model_path: str):
        """Load model weights.
        
        Args:
            model_path: Path to model checkpoint
        """
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
    
    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Make predictions on input data.
        
        Args:
            X: Input features (batch_size, seq_len, num_features)
        
        Returns:
            Tuple of (direction_probs, direction_pred, volatility_pred)
        """
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            dir_logits, vol_pred = self.model(X_tensor)
            direction_probs = torch.softmax(dir_logits, dim=1)
            direction_pred = torch.argmax(direction_probs, dim=1)
        
        return (
            direction_probs.cpu().numpy(),
            direction_pred.cpu().numpy(),
            vol_pred.cpu().numpy()
        )
    
    def predict_single(
        self,
        X: np.ndarray
    ) -> Dict[str, float]:
        """Make prediction for a single sample.
        
        Args:
            X: Single sample (seq_len, num_features)
        
        Returns:
            Dictionary with predictions
        """
        X_batch = np.expand_dims(X, axis=0)  # Add batch dimension
        dir_probs, dir_pred, vol_pred = self.predict(X_batch)
        
        direction_names = {0: 'DOWN', 1: 'NEUTRAL', 2: 'UP'}
        
        return {
            'direction': direction_names[int(dir_pred[0])],
            'direction_confidence': float(dir_probs[0, int(dir_pred[0])]),
            'prob_down': float(dir_probs[0, 0]),
            'prob_neutral': float(dir_probs[0, 1]),
            'prob_up': float(dir_probs[0, 2]),
            'volatility': float(vol_pred[0, 0]),
        }


class SignalGenerator:
    """Generate trading signals from model predictions."""
    
    def __init__(
        self,
        confidence_threshold: float = 0.6,
        volatility_threshold: float = 0.02,
    ):
        """Initialize signal generator.
        
        Args:
            confidence_threshold: Minimum confidence for signal generation
            volatility_threshold: Maximum acceptable volatility for signal
        """
        self.confidence_threshold = confidence_threshold
        self.volatility_threshold = volatility_threshold
    
    def generate_signal(
        self,
        prediction: Dict[str, float]
    ) -> Optional[Dict]:
        """Generate trading signal from prediction.
        
        Args:
            prediction: Dictionary from Predictor.predict_single
        
        Returns:
            Signal dictionary or None if no signal
        """
        direction = prediction['direction']
        confidence = prediction['direction_confidence']
        volatility = prediction['volatility']
        
        # Check if confidence is above threshold
        if confidence < self.confidence_threshold:
            logger.debug(f"Signal rejected: confidence {confidence:.3f} < {self.confidence_threshold}")
            return None
        
        # Check if volatility is acceptable
        if volatility > self.volatility_threshold:
            logger.debug(f"Signal rejected: volatility {volatility:.4f} > {self.volatility_threshold}")
            return None
        
        # Skip NEUTRAL signals
        if direction == 'NEUTRAL':
            return None
        
        signal = {
            'action': 'BUY' if direction == 'UP' else 'SELL',
            'direction': direction,
            'confidence': confidence,
            'volatility': volatility,
            'strength': self._calculate_signal_strength(confidence, volatility),
        }
        
        return signal
    
    @staticmethod
    def _calculate_signal_strength(confidence: float, volatility: float) -> float:
        """Calculate overall signal strength (0-1).
        
        Args:
            confidence: Model confidence
            volatility: Predicted volatility
        
        Returns:
            Signal strength score
        """
        # Higher confidence = stronger signal
        # Lower volatility = stronger signal
        strength = confidence * (1 - volatility)
        return max(0, min(1, strength))
    
    def generate_signals_batch(
        self,
        predictions: list
    ) -> list:
        """Generate signals for multiple predictions.
        
        Args:
            predictions: List of prediction dictionaries
        
        Returns:
            List of signal dictionaries (excludes None values)
        """
        signals = []
        for pred in predictions:
            signal = self.generate_signal(pred)
            if signal:
                signals.append(signal)
        return signals


class InferenceEngine:
    """Complete inference engine combining predictor and signal generation."""
    
    def __init__(
        self,
        model,
        confidence_threshold: float = 0.6,
        volatility_threshold: float = 0.02,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        """Initialize inference engine.
        
        Args:
            model: Trained model
            confidence_threshold: Signal generation threshold
            volatility_threshold: Volatility threshold
            device: Device to use
        """
        self.predictor = Predictor(model, device=device)
        self.signal_generator = SignalGenerator(
            confidence_threshold=confidence_threshold,
            volatility_threshold=volatility_threshold,
        )
    
    def run(
        self,
        X: np.ndarray,
        symbol: str = 'UNKNOWN',
        timeframe: str = '1h',
    ) -> Dict:
        """Run complete inference pipeline.
        
        Args:
            X: Input features
            symbol: Trading symbol
            timeframe: Timeframe
        
        Returns:
            Result dictionary with predictions and signals
        """
        if X.ndim == 2:
            # Single sample
            prediction = self.predictor.predict_single(X)
            signal = self.signal_generator.generate_signal(prediction)
        else:
            # Batch
            dir_probs, dir_pred, vol_pred = self.predictor.predict(X)
            # Use last sample
            prediction = {
                'direction': ['DOWN', 'NEUTRAL', 'UP'][int(dir_pred[-1])],
                'direction_confidence': float(dir_probs[-1, int(dir_pred[-1])]),
                'prob_down': float(dir_probs[-1, 0]),
                'prob_neutral': float(dir_probs[-1, 1]),
                'prob_up': float(dir_probs[-1, 2]),
                'volatility': float(vol_pred[-1, 0]),
            }
            signal = self.signal_generator.generate_signal(prediction)
        
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'prediction': prediction,
            'signal': signal,
            'has_signal': signal is not None,
        }


def main():
    """Example inference usage."""
    from src.models.model import create_model
    
    logger.info("Running inference example...")
    
    # Create model
    model = create_model('lstm', input_size=30)
    
    # Initialize inference engine
    engine = InferenceEngine(model)
    
    # Create dummy input
    X_test = np.random.randn(1, 20, 30).astype(np.float32)
    
    # Run inference
    result = engine.run(X_test, symbol='BTCUSDT', timeframe='1h')
    
    logger.info(f"Result: {result}")
    if result['has_signal']:
        logger.info(f"Signal generated: {result['signal']}")


if __name__ == "__main__":
    main()
