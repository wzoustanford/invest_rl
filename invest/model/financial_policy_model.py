"""
Financial Policy Model for DQN/R2D2 integration.
This model adapts the financial data and implements the policy for trading decisions.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, Tuple, Optional

# Add angle/RL to path to import device utilities
sys.path.append('/home/ubuntu/code/angle/RL')

# Import with fallback
try:
    from model.device_utils import get_device_manager
except ImportError:
    # Fallback implementation
    class MockDeviceManager:
        def __init__(self, device=None):
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        def to_dev(self, x):
            return x.to(self.device) if hasattr(x, 'to') else x
    
    def get_device_manager(device=None):
        return MockDeviceManager(device)


class FinancialPolicyNetwork(nn.Module):
    """
    Deep Q-Network specifically designed for financial trading.
    Takes market features and portfolio state as input.
    """
    
    def __init__(self, 
                 input_dim: int,
                 n_actions: int,
                 hidden_dims: list = [512, 256, 128],
                 use_dueling: bool = True,
                 dropout_rate: float = 0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_actions = n_actions
        self.use_dueling = use_dueling
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            # Only add BatchNorm for training stability (can be disabled for single obs)
            if i < len(hidden_dims) - 1:  # Don't add to last layer
                layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        if use_dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_head = nn.Linear(prev_dim, 1)
            self.advantage_head = nn.Linear(prev_dim, n_actions)
        else:
            # Standard DQN
            self.q_head = nn.Linear(prev_dim, n_actions)
    
    def forward(self, x):
        """Forward pass through the network"""
        features = self.feature_extractor(x)
        
        if self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard DQN
            q_values = self.q_head(features)
        
        return q_values


class FinancialPolicyModel:
    """
    Main policy model for financial trading that interfaces with DQN/R2D2.
    Handles data preprocessing, action selection, and reward computation.
    """
    
    def __init__(self,
                 data_list_file: str,
                 ticker_hash_file: str,
                 feature_dim: int = 249,
                 max_stocks: int = 100,
                 n_action_bins: int = 21,
                 device: str = None,
                 use_dueling: bool = True):
        
        self.devmgr = get_device_manager(device)
        self.device = self.devmgr.device
        
        # Load data configuration
        self.data_list_file = data_list_file
        self.ticker_hash_file = ticker_hash_file
        self.feature_dim = feature_dim
        self.max_stocks = max_stocks
        self.n_action_bins = n_action_bins
        
        # Load ticker hash
        import pickle
        with open(ticker_hash_file, 'rb') as f:
            self.ticker_hash = pickle.load(f)
        
        self.num_stocks = self.ticker_hash['num_tickers']
        
        # Calculate observation dimension
        # Features (249) + Portfolio state (min(num_stocks, 100)) + Scalars (5)
        self.obs_dim = feature_dim + min(self.num_stocks, max_stocks) + 5
        
        # Action space: simplified discrete actions
        # Each action represents a trading strategy
        self.n_actions = min(self.num_stocks * n_action_bins, 5000)  # Cap for feasibility
        
        # Initialize policy network
        self.policy_net = FinancialPolicyNetwork(
            input_dim=self.obs_dim,
            n_actions=self.n_actions,
            use_dueling=use_dueling
        ).to(self.device)
        
        print(f"Financial Policy Model initialized:")
        print(f"  Input dim: {self.obs_dim}")
        print(f"  Action space: {self.n_actions}")
        print(f"  Stocks: {self.num_stocks}")
        print(f"  Device: {self.device}")
    
    def process_observation(self, 
                          features: torch.Tensor, 
                          tickers: list,
                          portfolio_state: Optional[torch.Tensor] = None,
                          market_scalars: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process raw financial data into observation suitable for DQN.
        
        Args:
            features: Market features tensor (n_stocks, feature_dim)
            tickers: List of ticker symbols
            portfolio_state: Current portfolio weights
            market_scalars: Additional market information
        
        Returns:
            Processed observation tensor
        """
        # Map tickers to unified space
        unified_features = torch.zeros((self.num_stocks, self.feature_dim))
        portfolio_vec = torch.zeros(min(self.num_stocks, self.max_stocks))
        
        hash_dict = self.ticker_hash['hash_D']
        valid_stocks = 0
        
        for i, ticker in enumerate(tickers):
            if ticker in hash_dict and i < len(features):
                unified_idx = hash_dict[ticker]
                if unified_idx < self.num_stocks:
                    unified_features[unified_idx] = features[i]
                    valid_stocks += 1
                    
                    # Portfolio state
                    if (portfolio_state is not None and 
                        unified_idx < self.max_stocks and 
                        unified_idx < len(portfolio_state)):
                        portfolio_vec[unified_idx] = portfolio_state[unified_idx]
        
        # Aggregate features (mean of valid stocks)
        if valid_stocks > 0:
            aggregated_features = torch.mean(unified_features[:valid_stocks], dim=0)
        else:
            aggregated_features = torch.zeros(self.feature_dim)
        
        # Market scalars (portfolio value, previous returns, etc.)
        if market_scalars is None:
            market_scalars = torch.zeros(5)
        elif len(market_scalars) < 5:
            market_scalars = torch.cat([
                market_scalars, 
                torch.zeros(5 - len(market_scalars))
            ])
        
        # Combine all components
        observation = torch.cat([
            aggregated_features,
            portfolio_vec,
            market_scalars[:5]
        ])
        
        return observation.float()
    
    def select_action(self, 
                     observation: torch.Tensor, 
                     epsilon: float = 0.0) -> int:
        """
        Select trading action using epsilon-greedy policy.
        
        Args:
            observation: Processed market observation
            epsilon: Exploration rate
        
        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        with torch.no_grad():
            obs_batch = observation.unsqueeze(0).to(self.device)
            q_values = self.policy_net(obs_batch)
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def decode_action(self, action: int, num_focus_stocks: int = 20) -> torch.Tensor:
        """
        Convert discrete action to portfolio allocation.
        
        Args:
            action: Discrete action index
            num_focus_stocks: Number of stocks to focus on
        
        Returns:
            Portfolio weights tensor
        """
        portfolio = torch.zeros(self.num_stocks)
        
        if self.num_stocks <= 10:
            # Small universe: direct mapping
            stock_idx = action % self.num_stocks
            weight_idx = action // self.num_stocks
            weight = weight_idx / (self.n_action_bins - 1)
            portfolio[stock_idx] = weight
        else:
            # Large universe: focus strategy
            num_focus = min(num_focus_stocks, self.num_stocks)
            
            # Map action to focused stocks
            focus_action = action % (num_focus * self.n_action_bins)
            stock_idx = focus_action % num_focus
            weight_idx = focus_action // num_focus
            
            # Primary allocation
            primary_weight = weight_idx / (self.n_action_bins - 1)
            portfolio[stock_idx] = primary_weight
            
            # Distribute remaining weight
            remaining_weight = 1.0 - primary_weight
            if remaining_weight > 0:
                other_stocks = min(5, num_focus - 1)
                if other_stocks > 0:
                    other_weight = remaining_weight / other_stocks
                    for i in range(other_stocks):
                        if i != stock_idx:
                            portfolio[i] = other_weight
        
        # Normalize portfolio
        total_weight = torch.sum(portfolio)
        if total_weight > 0:
            portfolio = portfolio / total_weight
        else:
            portfolio = torch.ones(self.num_stocks) / self.num_stocks
        
        return portfolio
    
    def compute_reward(self, 
                      prices: torch.Tensor,
                      tickers: list,
                      portfolio: torch.Tensor,
                      previous_value: float = 1.0) -> Tuple[float, Dict]:
        """
        Compute trading reward (Sharpe ratio based).
        
        Args:
            prices: Price data (n_stocks, time_steps)
            tickers: Ticker symbols
            portfolio: Portfolio weights
            previous_value: Previous portfolio value
        
        Returns:
            Reward value and info dict
        """
        if prices.shape[1] < 2:
            return 0.0, {'sharpe': 0.0, 'return': 0.0}
        
        # Map prices to unified space
        unified_prices = torch.zeros((self.num_stocks, prices.shape[1]))
        hash_dict = self.ticker_hash['hash_D']
        
        for i, ticker in enumerate(tickers):
            if ticker in hash_dict and i < len(prices):
                unified_idx = hash_dict[ticker]
                if unified_idx < self.num_stocks:
                    unified_prices[unified_idx] = prices[i]
        
        # Compute returns
        initial_prices = unified_prices[:, 0] + 1e-10
        final_prices = unified_prices[:, -1]
        
        # Portfolio performance
        portfolio_shares = portfolio / initial_prices
        period_return = torch.sum((final_prices - initial_prices) * portfolio_shares)
        
        # Time series for Sharpe calculation
        returns_series = []
        for t in range(1, unified_prices.shape[1]):
            step_return = torch.sum(
                (unified_prices[:, t] - unified_prices[:, t-1]) * portfolio_shares
            )
            returns_series.append(step_return.item())
        
        if len(returns_series) > 1:
            mean_return = np.mean(returns_series)
            std_return = np.std(returns_series) + 1e-8
            sharpe = mean_return / std_return
        else:
            sharpe = 0.0
            mean_return = period_return.item()
        
        # Reward is Sharpe ratio with bonuses/penalties
        reward = float(sharpe)
        
        # Performance bonuses/penalties
        portfolio_value = previous_value * (1 + period_return.item() / 100)
        if portfolio_value < 0.5:
            reward -= 5.0  # Penalty for major losses
        elif portfolio_value > 1.5:
            reward += 1.0  # Bonus for good performance
        
        info = {
            'sharpe': sharpe,
            'return': period_return.item(),
            'portfolio_value': portfolio_value,
            'mean_return': mean_return
        }
        
        return reward, info
    
    def get_network_parameters(self):
        """Get policy network parameters for DQN training"""
        return self.policy_net.parameters()
    
    def get_network_state_dict(self):
        """Get policy network state dict for saving/loading"""
        return self.policy_net.state_dict()
    
    def load_network_state_dict(self, state_dict):
        """Load policy network state dict"""
        self.policy_net.load_state_dict(state_dict)
    
    def set_training_mode(self, training: bool = True):
        """Set network to training or evaluation mode"""
        self.policy_net.train(training)
    
    def to_device(self, tensor_or_model):
        """Move tensor or model to device"""
        return self.devmgr.to_dev(tensor_or_model)


def create_financial_policy_model(data_list_file: str,
                                ticker_hash_file: str,
                                device: str = None,
                                **kwargs) -> FinancialPolicyModel:
    """
    Factory function to create financial policy model.
    
    Args:
        data_list_file: Path to data list file
        ticker_hash_file: Path to ticker hash file
        device: Device to use ('cuda' or 'cpu')
        **kwargs: Additional arguments for FinancialPolicyModel
    
    Returns:
        Initialized FinancialPolicyModel
    """
    return FinancialPolicyModel(
        data_list_file=data_list_file,
        ticker_hash_file=ticker_hash_file,
        device=device,
        **kwargs
    )