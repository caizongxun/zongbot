"""Neural Network Models for Price Prediction - Phase 3

Implements LSTM/GRU architectures for predicting price direction and volatility.
"""

import logging
from typing import Tuple, Optional

import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class LSTMPredictor(nn.Module):
    """LSTM-based model for predicting price direction and volatility.
    
    Architecture:
    - Input: Multiple technical indicators
    - Hidden: 2-3 LSTM layers with dropout
    - Output: Direction (classification) + Volatility (regression)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size_direction: int = 3,  # Down, Neutral, Up
        output_size_volatility: int = 1,
    ):
        """Initialize LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_size_direction: Output size for direction classification (3: Down, Neutral, Up)
            output_size_volatility: Output size for volatility prediction
        """
        super(LSTMPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size_direction = output_size_direction
        self.output_size_volatility = output_size_volatility
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Direction prediction head (classification)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size_direction)
        )
        
        # Volatility prediction head (regression)
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size_volatility),
            nn.ReLU()  # Volatility is always positive
        )
        
        logger.info(f"LSTMPredictor initialized: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Tuple of (direction_logits, volatility_pred)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: (batch, seq_len, hidden)
        
        # Take last hidden state
        last_hidden = lstm_out[:, -1, :]  # (batch, hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Direction prediction
        direction_logits = self.direction_head(last_hidden)  # (batch, 3)
        
        # Volatility prediction
        volatility_pred = self.volatility_head(last_hidden)  # (batch, 1)
        
        return direction_logits, volatility_pred


class GRUPredictor(nn.Module):
    """GRU-based model for predicting price direction and volatility.
    
    Similar to LSTM but with GRU cells (fewer parameters).
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size_direction: int = 3,
        output_size_volatility: int = 1,
    ):
        """Initialize GRU model.
        
        Args:
            input_size: Number of input features
            hidden_size: GRU hidden dimension
            num_layers: Number of GRU layers
            dropout: Dropout rate
            output_size_direction: Output size for direction classification
            output_size_volatility: Output size for volatility prediction
        """
        super(GRUPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size_direction = output_size_direction
        self.output_size_volatility = output_size_volatility
        
        # GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Direction prediction head
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size_direction)
        )
        
        # Volatility prediction head
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size_volatility),
            nn.ReLU()
        )
        
        logger.info(f"GRUPredictor initialized: input={input_size}, hidden={hidden_size}, layers={num_layers}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
        
        Returns:
            Tuple of (direction_logits, volatility_pred)
        """
        # GRU forward
        gru_out, h_n = self.gru(x)  # gru_out: (batch, seq_len, hidden)
        
        # Take last hidden state
        last_hidden = gru_out[:, -1, :]  # (batch, hidden)
        last_hidden = self.dropout(last_hidden)
        
        # Direction prediction
        direction_logits = self.direction_head(last_hidden)
        
        # Volatility prediction
        volatility_pred = self.volatility_head(last_hidden)
        
        return direction_logits, volatility_pred


class AttentionLSTM(nn.Module):
    """LSTM with Attention mechanism for better feature extraction."""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        output_size_direction: int = 3,
        output_size_volatility: int = 1,
    ):
        """Initialize Attention-LSTM model."""
        super(AttentionLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size_direction)
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_size_volatility),
            nn.ReLU()
        )
        
        logger.info(f"AttentionLSTM initialized: input={input_size}, hidden={hidden_size}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention."""
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Multi-head attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        attn_out = attn_out + lstm_out
        attn_out = self.dropout(attn_out)
        
        # Take last output
        last_output = attn_out[:, -1, :]
        
        # Predictions
        direction_logits = self.direction_head(last_output)
        volatility_pred = self.volatility_head(last_output)
        
        return direction_logits, volatility_pred


def create_model(
    model_type: str,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """Factory function to create models.
    
    Args:
        model_type: Type of model ('lstm', 'gru', 'attention')
        input_size: Number of input features
        hidden_size: Hidden dimension
        num_layers: Number of layers
        dropout: Dropout rate
        device: Device to use
    
    Returns:
        Model instance on specified device
    """
    if model_type.lower() == 'lstm':
        model = LSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type.lower() == 'gru':
        model = GRUPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    elif model_type.lower() == 'attention':
        model = AttentionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)
