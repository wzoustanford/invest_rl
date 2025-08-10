"""
LSTM-based Adaptive Trading Model
Features:
1. LSTM for temporal pattern learning
2. Dynamic gamma prediction
3. Trade timing decision (trade now vs wait)
4. Ensemble predictions for stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class LSTMAdaptiveModel(nn.Module):
    """
    LSTM model with multiple heads:
    - Portfolio allocation head
    - Gamma prediction head  
    - Trade timing head
    """
    
    def __init__(self, 
                 input_dim=None,
                 lstm_hidden_dim=128,
                 lstm_layers=2,
                 dropout_rate=0.2,
                 num_stocks=None,
                 sequence_length=7):
        """
        Args:
            input_dim: Feature dimension per timestep
            lstm_hidden_dim: Hidden dimension for LSTM
            lstm_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            num_stocks: Number of stocks to allocate
            sequence_length: Length of input sequence
        """
        super(LSTMAdaptiveModel, self).__init__()
        
        self.input_dim = input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.num_stocks = num_stocks
        self.sequence_length = sequence_length
        
        # LSTM backbone - bidirectional for better context
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism for importance weighting
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.Tanh(),
            nn.Linear(lstm_hidden_dim, 1)
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Portfolio allocation head
        self.portfolio_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, num_stocks)
        )
        
        # Gamma prediction head (predicts optimal gamma value)
        self.gamma_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # Trade timing head (binary: trade now or wait)
        self.timing_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, 2)  # 2 classes: trade/wait
        )
        
        # Market regime classifier (bull/bear/sideways)
        self.regime_head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden_dim // 2, 3)  # 3 regimes
        )
        
    def forward(self, x, return_all_heads=False):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            return_all_heads: If True, return outputs from all heads
            
        Returns:
            If return_all_heads=False: portfolio weights
            If return_all_heads=True: dict with all outputs
        """
        batch_size = x.size(0)
        
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, lstm_hidden_dim * 2)
        
        # Apply attention mechanism
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)
        # attention_weights shape: (batch_size, sequence_length, 1)
        
        # Weighted sum of LSTM outputs
        attended = torch.sum(lstm_out * attention_weights, dim=1)
        # attended shape: (batch_size, lstm_hidden_dim * 2)
        
        # Apply dropout
        attended = self.dropout(attended)
        
        # Generate outputs from each head
        portfolio_weights = F.softmax(self.portfolio_head(attended), dim=-1)
        gamma_value = self.gamma_head(attended).squeeze(-1)
        timing_logits = self.timing_head(attended)
        regime_logits = self.regime_head(attended)
        
        if return_all_heads:
            return {
                'portfolio_weights': portfolio_weights,
                'gamma_value': gamma_value,
                'timing_logits': timing_logits,
                'timing_probs': F.softmax(timing_logits, dim=-1),
                'regime_logits': regime_logits,
                'regime_probs': F.softmax(regime_logits, dim=-1),
                'attention_weights': attention_weights.squeeze(-1)
            }
        else:
            return portfolio_weights


class LSTMEnsemble(nn.Module):
    """
    Ensemble of LSTM models for more stable predictions
    """
    
    def __init__(self, 
                 num_models=3,
                 input_dim=None,
                 lstm_hidden_dims=[64, 128, 256],
                 num_stocks=None):
        """
        Args:
            num_models: Number of models in ensemble
            input_dim: Feature dimension
            lstm_hidden_dims: List of hidden dimensions for each model
            num_stocks: Number of stocks
        """
        super(LSTMEnsemble, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList()
        
        # Create diverse models
        for i in range(num_models):
            hidden_dim = lstm_hidden_dims[i % len(lstm_hidden_dims)]
            layers = 2 if i % 2 == 0 else 3
            dropout = 0.1 + (i * 0.1)
            
            model = LSTMAdaptiveModel(
                input_dim=input_dim,
                lstm_hidden_dim=hidden_dim,
                lstm_layers=layers,
                dropout_rate=dropout,
                num_stocks=num_stocks
            )
            self.models.append(model)
            
        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
    def forward(self, x, return_all_heads=False):
        """
        Forward pass through ensemble
        """
        all_outputs = []
        
        for model in self.models:
            output = model(x, return_all_heads=return_all_heads)
            all_outputs.append(output)
        
        # Apply softmax to ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        if return_all_heads:
            # Aggregate all outputs
            ensemble_output = {}
            for key in all_outputs[0].keys():
                if 'weights' in key or 'probs' in key:
                    # Weighted average for probabilities
                    stacked = torch.stack([out[key] for out in all_outputs])
                    ensemble_output[key] = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
                elif 'logits' in key:
                    # Average logits
                    stacked = torch.stack([out[key] for out in all_outputs])
                    ensemble_output[key] = torch.mean(stacked, dim=0)
                else:
                    # Weighted average for other outputs
                    stacked = torch.stack([out[key] for out in all_outputs])
                    if len(stacked.shape) == 3:
                        ensemble_output[key] = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
                    else:
                        ensemble_output[key] = torch.sum(stacked * weights.view(-1, 1), dim=0)
            return ensemble_output
        else:
            # Simple weighted average for portfolio weights
            stacked = torch.stack(all_outputs)
            return torch.sum(stacked * weights.view(-1, 1, 1), dim=0)


class AdaptiveLSTMTrainer:
    """
    Training utilities for the LSTM model with adaptive features
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_with_adaptive_loss(self, 
                                  data_loader, 
                                  optimizer,
                                  num_epochs=100,
                                  gamma_weight=0.1,
                                  timing_weight=0.1,
                                  regime_weight=0.05):
        """
        Train with multi-task learning
        
        Args:
            data_loader: DataLoader with training data
            optimizer: Optimizer
            num_epochs: Number of training epochs
            gamma_weight: Weight for gamma prediction loss
            timing_weight: Weight for timing prediction loss
            regime_weight: Weight for regime prediction loss
        """
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_losses = []
            
            for batch in data_loader:
                features = batch['features'].to(self.device)
                returns = batch['returns'].to(self.device)
                optimal_gamma = batch.get('optimal_gamma', None)
                should_trade = batch.get('should_trade', None)
                market_regime = batch.get('market_regime', None)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(features, return_all_heads=True)
                
                # Portfolio loss (Sharpe ratio based)
                portfolio_weights = outputs['portfolio_weights']
                portfolio_returns = torch.sum(portfolio_weights * returns, dim=-1)
                sharpe_ratio = portfolio_returns / (torch.std(portfolio_returns) + 1e-8)
                portfolio_loss = -sharpe_ratio.mean()
                
                total_loss = portfolio_loss
                
                # Gamma prediction loss (if labels available)
                if optimal_gamma is not None:
                    gamma_loss = F.mse_loss(outputs['gamma_value'], optimal_gamma)
                    total_loss += gamma_weight * gamma_loss
                
                # Timing loss (if labels available)
                if should_trade is not None:
                    timing_loss = F.cross_entropy(outputs['timing_logits'], should_trade)
                    total_loss += timing_weight * timing_loss
                
                # Regime loss (if labels available)
                if market_regime is not None:
                    regime_loss = F.cross_entropy(outputs['regime_logits'], market_regime)
                    total_loss += regime_weight * regime_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            if (epoch + 1) % 10 == 0:
                avg_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    def predict_with_confidence(self, features):
        """
        Make predictions with confidence estimates
        
        Args:
            features: Input features
            
        Returns:
            Dictionary with predictions and confidence scores
        """
        self.model.eval()
        
        with torch.no_grad():
            features = features.to(self.device)
            
            # Run multiple forward passes with dropout for uncertainty
            n_samples = 10
            all_predictions = []
            
            for _ in range(n_samples):
                pred = self.model(features, return_all_heads=True)
                all_predictions.append(pred)
            
            # Aggregate predictions
            final_predictions = {}
            
            # Portfolio weights - mean and std
            portfolio_weights = torch.stack([p['portfolio_weights'] for p in all_predictions])
            final_predictions['portfolio_weights'] = portfolio_weights.mean(dim=0)
            final_predictions['portfolio_confidence'] = 1.0 / (1.0 + portfolio_weights.std(dim=0).mean())
            
            # Gamma - mean and std
            gamma_values = torch.stack([p['gamma_value'] for p in all_predictions])
            final_predictions['gamma_value'] = gamma_values.mean(dim=0)
            final_predictions['gamma_confidence'] = 1.0 / (1.0 + gamma_values.std(dim=0))
            
            # Timing - probability of trading
            timing_probs = torch.stack([p['timing_probs'] for p in all_predictions])
            final_predictions['should_trade'] = timing_probs.mean(dim=0)[:, 0] > 0.5
            final_predictions['trade_confidence'] = timing_probs.mean(dim=0).max(dim=-1)[0]
            
            # Regime - most likely regime
            regime_probs = torch.stack([p['regime_probs'] for p in all_predictions])
            final_predictions['regime_probs'] = regime_probs.mean(dim=0)
            final_predictions['market_regime'] = regime_probs.mean(dim=0).argmax(dim=-1)
            
            return final_predictions