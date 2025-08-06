"""
Optimized Sequential Supervised Learning Trainer

This module implements a supervised learning approach for portfolio optimization
using sequential rewards discounted by gamma. Key optimizations:
- GPU memory management with gradient accumulation
- Batch processing for efficiency  
- Mixed precision training support
- Memory-efficient data loading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import gc


@dataclass
class TrainingConfig:
    """Configuration for sequential supervised training."""
    gamma: float = 0.3
    learning_rate: float = 0.001
    num_steps: int = 750
    num_consecutive_days: int = 7  # Number of sequential days to consider
    obj_use_mean_return: bool = False  # Use actual return for Sharpe
    gradient_accumulation_steps: int = 4  # For memory efficiency
    use_mixed_precision: bool = True
    log_interval: int = 50
    eval_interval: int = 50
    device: str = 'cuda'
    seed: int = 42
    dropout_ratio: float = 0.0
    num_conv_filters: int = 32
    hidden_dim: int = 47


class OptimizedIIModel(nn.Module):
    """Optimized version of IIMODEL with memory efficiency improvements."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.num_conv_filters = config.num_conv_filters
        self.hidden_dim = config.hidden_dim
        self.adaptive_max_pool_output = 10
        
        # Convolutional layers with memory-efficient settings
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.num_conv_filters, kernel_size=3),
            nn.Softplus(),
            nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            nn.Flatten(1, 2)
        )
        
        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_conv_filters * self.adaptive_max_pool_output, self.hidden_dim),
            nn.Tanh(),
        )
        
        self.fc1_dropout = nn.Dropout(p=config.dropout_ratio)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        self.sm = nn.Softmax(dim=0)
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing = False
    
    def forward(self, x, return_acts=False):
        # Add channel dimension
        x = torch.unsqueeze(x, 1)
        
        # Normalize by max (memory efficient)
        with torch.no_grad():
            ma = torch.max(x, dim=2, keepdim=True)[0]
            x = x / (ma + 1e-10)
        
        # Forward pass
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        acts = x
        x = self.fc2(x)
        x = self.sm(x)
        
        if return_acts:
            return x, acts
        return x


class SequentialSupervisedTrainer:
    """Main trainer class for sequential supervised learning."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        # Initialize model
        self.model = OptimizedIIModel(config).to(self.device)
        
        # Optimizer with gradient clipping
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=1e-5  # Add L2 regularization
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.use_mixed_precision else None
        
        # Training history
        self.history = {
            'loss': [],
            'eval_sharpe': [],
            'eval_return': [],
            'eval_stddev': []
        }
    
    def compute_sharpe_ratio(self, 
                            series: torch.Tensor, 
                            portfolio_weights: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Compute Sharpe ratio and related metrics efficiently."""
        
        # Compute portfolio shares
        initial_prices = series[:, 0:1] + 1e-10
        portfolio_shares = portfolio_weights / initial_prices
        
        # Compute returns
        price_changes = series[:, 1:] - series[:, 0:1]
        returns_series = torch.sum(price_changes * portfolio_shares, dim=0)
        
        # Final return
        final_prices = series[:, -1:] 
        actual_return = torch.sum((final_prices - series[:, 0:1]) * portfolio_shares)
        
        # Statistics
        mean_return = torch.mean(returns_series)
        stddev = torch.std(returns_series) + 1e-10
        
        # Sharpe ratio
        if self.config.obj_use_mean_return:
            sharpe = mean_return / stddev
        else:
            sharpe = actual_return / stddev
        
        return sharpe, mean_return, actual_return, stddev
    
    def train_step(self, data_sequence: List[Dict]) -> float:
        """Single training step with gradient accumulation."""
        
        self.model.train()
        total_loss = 0.0
        T = len(data_sequence)
        
        # Gradient accumulation for memory efficiency
        accumulation_steps = min(self.config.gradient_accumulation_steps, T)
        
        for acc_step in range(0, T, accumulation_steps):
            batch_loss = torch.tensor(0.0, device=self.device)
            
            for i in range(acc_step, min(acc_step + accumulation_steps, T)):
                # Load data to GPU only when needed
                features = data_sequence[i]['trainFeature'].to(self.device, non_blocking=True)
                series = data_sequence[i]['train_in_portfolio_series'].to(self.device, non_blocking=True)
                
                # Mixed precision forward pass
                if self.config.use_mixed_precision:
                    with autocast():
                        portfolio_weights = self.model(features)
                        sharpe, _, _, _ = self.compute_sharpe_ratio(series, portfolio_weights)
                        
                        # Discounted loss
                        gamma_power = self.config.gamma ** (T - i - 1)
                        loss = -sharpe * gamma_power
                else:
                    portfolio_weights = self.model(features)
                    sharpe, _, _, _ = self.compute_sharpe_ratio(series, portfolio_weights)
                    gamma_power = self.config.gamma ** (T - i - 1)
                    loss = -sharpe * gamma_power
                
                batch_loss = batch_loss + loss / accumulation_steps
                
                # Free memory
                del features, series
                torch.cuda.empty_cache()
            
            # Backward pass with gradient scaling
            if self.config.use_mixed_precision:
                self.scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()
            
            total_loss += batch_loss.item() * accumulation_steps
        
        # Optimizer step with gradient clipping
        if self.config.use_mixed_precision:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        return total_loss
    
    def evaluate(self, test_data: Dict) -> Dict[str, float]:
        """Evaluate model on test data."""
        
        self.model.eval()
        
        with torch.no_grad():
            # Check if test data exists
            if test_data.get('test_in_portfolio_series') is None:
                return {}
            
            features = test_data['testFeature'].to(self.device)
            series = test_data['test_in_portfolio_series'].to(self.device)
            
            # Forward pass
            portfolio_weights = self.model(features)
            
            # Compute metrics
            sharpe, mean_return, actual_return, stddev = self.compute_sharpe_ratio(
                series, portfolio_weights
            )
            
            # Get top stocks
            top_k = min(20, len(portfolio_weights))
            top_weights, top_indices = torch.topk(portfolio_weights.squeeze(), top_k)
            
            top_stocks = []
            if 'all_test_tickers' in test_data:
                for idx in top_indices:
                    top_stocks.append(test_data['all_test_tickers'][idx])
        
        return {
            'sharpe': sharpe.item(),
            'mean_return': mean_return.item(),
            'actual_return': actual_return.item(),
            'stddev': stddev.item(),
            'top_stocks': top_stocks,
            'portfolio_weights': portfolio_weights.cpu().numpy()
        }
    
    def train_on_sequence(self, 
                         data_files: List[str],
                         save_dir: str,
                         model_name: str) -> Dict:
        """Train on a sequence of consecutive data files."""
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Load data sequence
        data_sequence = []
        for file_path in data_files:
            with open(file_path, 'rb') as f:
                data_sequence.append(pickle.load(f))
        
        # Training loop
        for step in range(1, self.config.num_steps + 1):
            # Training step
            loss = self.train_step(data_sequence)
            self.history['loss'].append(loss)
            
            # Logging
            if step % self.config.log_interval == 0:
                print(f'Step {step}/{self.config.num_steps} | Loss: {loss:.6f}')
            
            # Evaluation
            if step % self.config.eval_interval == 0:
                eval_results = self.evaluate(data_sequence[-1])
                
                if eval_results:
                    self.history['eval_sharpe'].append(eval_results['sharpe'])
                    self.history['eval_return'].append(eval_results['actual_return'])
                    self.history['eval_stddev'].append(eval_results['stddev'])
                    
                    print(f'  Eval | Sharpe: {eval_results["sharpe"]:.4f} | '
                          f'Return: {eval_results["actual_return"]:.4f} | '
                          f'Std: {eval_results["stddev"]:.4f}')
                
                # Save checkpoint
                if step % (self.config.eval_interval * 2) == 0:
                    checkpoint_path = os.path.join(save_dir, f'{model_name}_step{step}.pt')
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'step': step,
                        'history': self.history
                    }, checkpoint_path)
        
        # Save final model
        final_path = os.path.join(save_dir, f'{model_name}_final.pt')
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }, final_path)
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        gc.collect()
        
        return self.history
    
    def predict(self, features: torch.Tensor) -> np.ndarray:
        """Make predictions for portfolio allocation."""
        
        self.model.eval()
        with torch.no_grad():
            features = features.to(self.device)
            portfolio_weights = self.model(features)
            return portfolio_weights.cpu().numpy()