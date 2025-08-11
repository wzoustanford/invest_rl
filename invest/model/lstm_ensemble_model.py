"""
LSTM Ensemble Model
Combines 2-layer, 3-layer, and 4-layer LSTM architectures
Uses weighted averaging or dynamic weighting based on market conditions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from .lstm_sequential_model import LSTMSequentialModel, LSTMSequentialTrainer

class LSTMEnsemble(nn.Module):
    """
    Ensemble of LSTM models with different depths
    Combines predictions from 2L, 3L, and 4L architectures
    """
    
    def __init__(
        self,
        num_conv_filters: int = 32,
        hidden_dim: int = 47,
        lstm_hidden_dim: int = 64,
        dropout_ratio: float = 0.0,
        num_timesteps: int = 7,
        ensemble_layers: List[int] = [2, 3, 4],
        weighting_strategy: str = 'equal',  # 'equal', 'learned', 'adaptive', 'performance'
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.num_timesteps = num_timesteps
        self.ensemble_layers = ensemble_layers
        self.weighting_strategy = weighting_strategy
        self.device = device
        
        # Create individual LSTM models
        self.models = nn.ModuleList([
            LSTMSequentialModel(
                num_conv_filters=num_conv_filters,
                hidden_dim=hidden_dim,
                lstm_hidden_dim=lstm_hidden_dim,
                num_lstm_layers=num_layers,
                dropout_ratio=dropout_ratio,
                num_timesteps=num_timesteps
            )
            for num_layers in ensemble_layers
        ])
        
        # Initialize weights based on strategy
        if weighting_strategy == 'equal':
            # Equal weights for all models
            self.weights = torch.ones(len(ensemble_layers)) / len(ensemble_layers)
            self.weights = self.weights.to(device)
            
        elif weighting_strategy == 'learned':
            # Learnable weights (will be normalized with softmax)
            self.weight_params = nn.Parameter(torch.zeros(len(ensemble_layers)))
            
        elif weighting_strategy == 'adaptive':
            # Adaptive weights based on input features
            # Use a small network to predict weights from the last timestep features
            self.weight_predictor = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, len(ensemble_layers)),
                nn.Softmax(dim=-1)
            )
            
        elif weighting_strategy == 'performance':
            # Performance-based weights (updated during training)
            # Initialize with equal weights, will be updated based on validation performance
            self.register_buffer('performance_weights', 
                               torch.ones(len(ensemble_layers)) / len(ensemble_layers))
            self.register_buffer('performance_scores', 
                               torch.zeros(len(ensemble_layers)))
            
        # Optional: Temperature scaling for ensemble diversity
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(
        self, 
        features_sequence: List[torch.Tensor],
        return_all_timesteps: bool = False,
        return_individual_predictions: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through ensemble
        
        Args:
            features_sequence: List of feature tensors for each timestep
            return_all_timesteps: Whether to return predictions for all timesteps
            return_individual_predictions: Whether to also return individual model predictions
            
        Returns:
            Ensemble prediction (and optionally individual predictions)
        """
        
        # Get predictions from each model
        individual_predictions = []
        for model in self.models:
            pred = model(features_sequence, return_all_timesteps)
            individual_predictions.append(pred)
        
        # Stack predictions for easier manipulation
        # Shape: (num_models, num_stocks) or (num_models, num_timesteps, num_stocks)
        stacked_predictions = torch.stack(individual_predictions)
        
        # Determine weights based on strategy
        if self.weighting_strategy == 'equal':
            weights = self.weights.unsqueeze(-1)  # Add dimension for broadcasting
            if return_all_timesteps:
                weights = weights.unsqueeze(-1)  # Add another dimension for timesteps
                
        elif self.weighting_strategy == 'learned':
            weights = F.softmax(self.weight_params / self.temperature, dim=0)
            weights = weights.unsqueeze(-1)
            if return_all_timesteps:
                weights = weights.unsqueeze(-1)
                
        elif self.weighting_strategy == 'adaptive':
            # Use features from last timestep to predict weights
            last_features = features_sequence[-1]  # Shape: (num_stocks, hidden_dim)
            
            # Global pooling to get a single feature vector
            global_features = torch.mean(last_features, dim=0)  # Shape: (hidden_dim,)
            
            # Predict weights
            weights = self.weight_predictor(global_features)  # Shape: (num_models,)
            weights = weights.unsqueeze(-1)
            if return_all_timesteps:
                weights = weights.unsqueeze(-1)
                
        elif self.weighting_strategy == 'performance':
            weights = self.performance_weights.unsqueeze(-1)
            if return_all_timesteps:
                weights = weights.unsqueeze(-1)
        
        # Apply weights and sum
        ensemble_prediction = torch.sum(stacked_predictions * weights, dim=0)
        
        # Ensure weights sum to 1 (renormalize)
        ensemble_prediction = F.softmax(ensemble_prediction, dim=-1)
        
        if return_individual_predictions:
            return ensemble_prediction, individual_predictions
        else:
            return ensemble_prediction
    
    def update_performance_weights(self, model_idx: int, performance_score: float):
        """
        Update performance-based weights
        
        Args:
            model_idx: Index of the model
            performance_score: Performance metric (e.g., Sharpe ratio, return)
        """
        if self.weighting_strategy == 'performance':
            self.performance_scores[model_idx] = performance_score
            
            # Convert scores to weights using softmax
            # Add small epsilon to avoid division by zero
            positive_scores = self.performance_scores - self.performance_scores.min() + 1e-8
            self.performance_weights = positive_scores / positive_scores.sum()


class LSTMEnsembleTrainer:
    """
    Trainer for LSTM Ensemble
    Supports different training strategies
    """
    
    def __init__(
        self,
        ensemble_model: LSTMEnsemble,
        device: torch.device,
        gamma: float = 0.3,
        learning_rate: float = 0.001,
        training_strategy: str = 'joint',  # 'joint', 'sequential', 'independent'
        individual_lr: Optional[List[float]] = None
    ):
        self.model = ensemble_model.to(device)
        self.device = device
        self.gamma = gamma
        self.training_strategy = training_strategy
        
        # Set up optimizers based on strategy
        if training_strategy == 'joint':
            # Single optimizer for entire ensemble
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
            
        elif training_strategy == 'sequential':
            # Train models sequentially, each with its own optimizer
            self.optimizers = [
                torch.optim.Adam(model.parameters(), lr=learning_rate)
                for model in self.model.models
            ]
            self.current_model_idx = 0
            
        elif training_strategy == 'independent':
            # Train all models independently with different learning rates
            if individual_lr is None:
                individual_lr = [learning_rate] * len(ensemble_model.models)
            self.optimizers = [
                torch.optim.Adam(model.parameters(), lr=lr)
                for model, lr in zip(self.model.models, individual_lr)
            ]
    
    def compute_sharpe_loss(
        self,
        weights: torch.Tensor,
        price_series: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        Compute gamma-discounted Sharpe ratio loss
        """
        initial_prices = price_series[:, 0:1] + 1e-10
        shares = weights / initial_prices
        
        returns_matrix = (price_series[:, 1:] - price_series[:, :-1]) * shares.unsqueeze(1).expand(-1, price_series.shape[1]-1)
        
        period_returns = torch.sum(returns_matrix, dim=0)
        
        # Gamma discounting
        T = len(period_returns)
        gamma_weights = torch.tensor([gamma ** t for t in range(T)], 
                                    dtype=torch.float32, device=self.device)
        gamma_weights = gamma_weights / gamma_weights.sum()
        
        weighted_mean = torch.sum(period_returns * gamma_weights)
        weighted_variance = torch.sum(((period_returns - weighted_mean) ** 2) * gamma_weights)
        weighted_std = torch.sqrt(weighted_variance + 1e-10)
        
        sharpe_ratio = weighted_mean / weighted_std
        
        return -sharpe_ratio
    
    def train_step(self, data_sequence: List[Dict]) -> float:
        """
        Single training step
        """
        self.model.train()
        
        # Extract features and price series
        features_list = []
        for data in data_sequence:
            features = data['features'].to(self.device)
            features_list.append(features)
        
        price_series = data_sequence[-1]['price_series'].to(self.device)
        
        if self.training_strategy == 'joint':
            # Train ensemble jointly
            self.optimizer.zero_grad()
            
            # Get ensemble prediction
            weights = self.model(features_list, return_all_timesteps=False)
            
            # Compute loss
            loss = self.compute_sharpe_loss(weights, price_series, self.gamma)
            
            # Backward and optimize
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            
            self.optimizer.step()
            
            return loss.item()
            
        elif self.training_strategy == 'sequential':
            # Train one model at a time
            model_idx = self.current_model_idx
            optimizer = self.optimizers[model_idx]
            model = self.model.models[model_idx]
            
            optimizer.zero_grad()
            
            # Get prediction from current model
            weights = model(features_list, return_all_timesteps=False)
            
            # Compute loss
            loss = self.compute_sharpe_loss(weights, price_series, self.gamma)
            
            # Backward and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            
            # Move to next model
            self.current_model_idx = (self.current_model_idx + 1) % len(self.model.models)
            
            return loss.item()
            
        elif self.training_strategy == 'independent':
            # Train all models independently
            total_loss = 0.0
            
            for model, optimizer in zip(self.model.models, self.optimizers):
                optimizer.zero_grad()
                
                # Get prediction
                weights = model(features_list, return_all_timesteps=False)
                
                # Compute loss
                loss = self.compute_sharpe_loss(weights, price_series, self.gamma)
                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                total_loss += loss.item()
            
            return total_loss / len(self.model.models)
    
    def evaluate_individual_models(
        self,
        data_sequence: List[Dict]
    ) -> List[float]:
        """
        Evaluate each model individually to update performance weights
        """
        self.model.eval()
        performances = []
        
        features_list = []
        for data in data_sequence:
            features = data['features'].to(self.device)
            features_list.append(features)
        
        price_series = data_sequence[-1]['price_series'].to(self.device)
        
        with torch.no_grad():
            for i, model in enumerate(self.model.models):
                weights = model(features_list, return_all_timesteps=False)
                sharpe_loss = self.compute_sharpe_loss(weights, price_series, self.gamma)
                
                # Convert loss to performance score (negative loss = positive performance)
                performance = -sharpe_loss.item()
                performances.append(performance)
                
                # Update performance weights if using that strategy
                if self.model.weighting_strategy == 'performance':
                    self.model.update_performance_weights(i, performance)
        
        return performances