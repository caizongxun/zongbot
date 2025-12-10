"""Model Training Script - Phase 3

Training loop for LSTM/GRU models with validation and checkpointing.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class TimeSeriesDataset(torch.utils.data.Dataset):
    """Custom Dataset for time series prediction."""
    
    def __init__(
        self,
        X: np.ndarray,
        y_direction: np.ndarray,
        y_volatility: np.ndarray
    ):
        """Initialize dataset.
        
        Args:
            X: Features (N, seq_len, num_features)
            y_direction: Direction labels (N,) - 0: Down, 1: Neutral, 2: Up
            y_volatility: Volatility values (N,) - continuous values
        """
        self.X = torch.FloatTensor(X)
        self.y_direction = torch.LongTensor(y_direction)
        self.y_volatility = torch.FloatTensor(y_volatility)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y_direction[idx], self.y_volatility[idx]


class Trainer:
    """Model trainer with validation and checkpoint management."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'models/checkpoints',
        direction_weight: float = 1.0,
        volatility_weight: float = 0.5,
    ):
        """Initialize trainer.
        
        Args:
            model: Neural network model
            device: Device to use
            checkpoint_dir: Directory to save checkpoints
            direction_weight: Weight for direction classification loss
            volatility_weight: Weight for volatility regression loss
        """
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.direction_weight = direction_weight
        self.volatility_weight = volatility_weight
        
        # Loss functions
        self.criterion_direction = nn.CrossEntropyLoss()
        self.criterion_volatility = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        logger.info(f"Trainer initialized on device: {device}")
    
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Tuple[float, float, float]:
        """Train for one epoch.
        
        Args:
            train_loader: Training data loader
        
        Returns:
            Tuple of (avg_total_loss, avg_direction_loss, avg_volatility_loss)
        """
        self.model.train()
        total_loss = 0.0
        direction_loss_sum = 0.0
        volatility_loss_sum = 0.0
        
        for batch_X, batch_y_dir, batch_y_vol in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y_dir = batch_y_dir.to(self.device)
            batch_y_vol = batch_y_vol.to(self.device).unsqueeze(1)
            
            # Forward pass
            self.optimizer.zero_grad()
            dir_logits, vol_pred = self.model(batch_X)
            
            # Compute losses
            loss_dir = self.criterion_direction(dir_logits, batch_y_dir)
            loss_vol = self.criterion_volatility(vol_pred, batch_y_vol)
            loss = self.direction_weight * loss_dir + self.volatility_weight * loss_vol
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            direction_loss_sum += loss_dir.item()
            volatility_loss_sum += loss_vol.item()
        
        avg_total_loss = total_loss / len(train_loader)
        avg_direction_loss = direction_loss_sum / len(train_loader)
        avg_volatility_loss = volatility_loss_sum / len(train_loader)
        
        return avg_total_loss, avg_direction_loss, avg_volatility_loss
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, float, float, float]:
        """Validate model.
        
        Args:
            val_loader: Validation data loader
        
        Returns:
            Tuple of (avg_total_loss, direction_accuracy, volatility_mse, volatility_mae)
        """
        self.model.eval()
        total_loss = 0.0
        direction_correct = 0
        direction_total = 0
        volatility_mse_sum = 0.0
        volatility_mae_sum = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y_dir, batch_y_vol in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y_dir = batch_y_dir.to(self.device)
                batch_y_vol = batch_y_vol.to(self.device).unsqueeze(1)
                
                # Forward pass
                dir_logits, vol_pred = self.model(batch_X)
                
                # Compute losses
                loss_dir = self.criterion_direction(dir_logits, batch_y_dir)
                loss_vol = self.criterion_volatility(vol_pred, batch_y_vol)
                loss = self.direction_weight * loss_dir + self.volatility_weight * loss_vol
                
                total_loss += loss.item()
                
                # Direction accuracy
                _, predicted = torch.max(dir_logits, 1)
                direction_correct += (predicted == batch_y_dir).sum().item()
                direction_total += batch_y_dir.size(0)
                
                # Volatility metrics
                volatility_mse_sum += torch.mean((vol_pred - batch_y_vol) ** 2).item()
                volatility_mae_sum += torch.mean(torch.abs(vol_pred - batch_y_vol)).item()
        
        avg_loss = total_loss / len(val_loader)
        direction_accuracy = direction_correct / direction_total if direction_total > 0 else 0
        avg_volatility_mse = volatility_mse_sum / len(val_loader)
        avg_volatility_mae = volatility_mae_sum / len(val_loader)
        
        return avg_loss, direction_accuracy, avg_volatility_mse, avg_volatility_mae
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved at epoch {epoch} with val_loss={val_loss:.4f}")
    
    def load_checkpoint(self, checkpoint_path: str = 'best'):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint or 'best'/'latest'
        """
        if checkpoint_path in ['best', 'latest']:
            path = self.checkpoint_dir / f'{checkpoint_path}.pt'
        else:
            path = Path(checkpoint_path)
        
        if not path.exists():
            logger.warning(f"Checkpoint not found: {path}")
            return
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded from {path}")
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
    ):
        """Train model with validation and early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
        """
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Train
            train_loss, train_dir_loss, train_vol_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_dir_acc, val_vol_mse, val_vol_mae = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f} - "
                    f"Dir Acc: {val_dir_acc:.4f} - "
                    f"Vol MAE: {val_vol_mae:.4f}"
                )
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, is_best=True)
            else:
                self.patience_counter += 1
                self.save_checkpoint(epoch, val_loss)
            
            if self.patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        logger.info("Training complete!")
        self.load_checkpoint('best')


def main():
    """Example training script."""
    from src.models.model import create_model
    
    logger.info("Initializing model training...")
    
    # Create model
    model = create_model('lstm', input_size=30, hidden_size=128)
    
    # Create dummy data
    X_train = np.random.randn(1000, 20, 30).astype(np.float32)  # (N, seq_len, features)
    y_dir_train = np.random.randint(0, 3, 1000)  # Direction: 0, 1, 2
    y_vol_train = np.random.rand(1000).astype(np.float32)  # Volatility
    
    X_val = np.random.randn(200, 20, 30).astype(np.float32)
    y_dir_val = np.random.randint(0, 3, 200)
    y_vol_val = np.random.rand(200).astype(np.float32)
    
    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train, y_dir_train, y_vol_train)
    val_dataset = TimeSeriesDataset(X_val, y_dir_val, y_vol_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Train
    trainer = Trainer(model)
    trainer.fit(train_loader, val_loader, epochs=50, early_stopping_patience=10)


if __name__ == "__main__":
    main()
