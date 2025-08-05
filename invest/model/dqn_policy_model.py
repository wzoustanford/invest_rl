"""
DQN-compatible version of the original PolicyModel and ValueModel from angle_rl.
This replicates the original architecture but adapts it for DQN/R2D2 discrete action spaces.
"""

import torch
import torch.nn as nn
import torch.distributions as dd
import numpy as np
import sys
import pickle
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


class DQNPolicyNetwork(nn.Module):
    """
    DQN network that replicates the original PolicyModel architecture.
    Converts the continuous actor-critic policy to discrete Q-values.
    """
    
    def __init__(self, 
                 num_tickers: int,
                 num_actions: int,
                 dropout_ratio: float = 0.0,
                 num_conv_filters: int = 32,
                 hidden_dim: int = 47,
                 use_dueling: bool = True,
                 device=torch.device('cpu')):
        super().__init__()
        
        self.num_tickers = num_tickers
        self.num_actions = num_actions  
        self.hidden_dim = hidden_dim
        self.adaptive_max_pool_output = 10
        self.use_dueling = use_dueling
        self.device = device
        
        # Replicate original PolicyModel components
        self.prev_delta_net = nn.Sequential(
            nn.Linear(self.num_tickers, hidden_dim),
            nn.Tanh(),
        )
        self.prev_action_net = nn.Sequential(
            nn.Linear(self.num_tickers, hidden_dim),
            nn.Tanh(),
        )
        self.prev_sharpe_net = nn.Sequential(
            nn.Linear(1, hidden_dim), 
            nn.Tanh(),
        )
        
        # Feature convolution (from original PolicyModel)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=num_conv_filters, kernel_size=3),
            nn.Softplus(),
            nn.AdaptiveMaxPool1d(output_size=self.adaptive_max_pool_output),
            nn.Flatten(1, 2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(num_conv_filters * self.adaptive_max_pool_output, hidden_dim),
            nn.Softplus(),
        )
        self.fc1_dropout = nn.Dropout(p=dropout_ratio)
        
        # DQN output layers
        if use_dueling:
            # Dueling architecture: separate value and advantage streams
            self.value_head = nn.Linear(hidden_dim, 1)
            self.advantage_head = nn.Linear(hidden_dim, num_actions)
        else:
            # Standard DQN
            self.q_head = nn.Linear(hidden_dim, num_actions)
    
    def forward(self, state: Dict, tickers: list) -> torch.Tensor:
        """
        Forward pass replicating original PolicyModel structure.
        
        Args:
            state: Dict with keys 'delta', 'action', 'sharpe', 'features'
            tickers: List of ticker symbols
        
        Returns:
            Q-values for each action
        """
        # Process state components (from original PolicyModel)
        d = state['delta'].to(self.device)
        d = self.prev_delta_net(d)

        a = state['action'].to(self.device)  
        a = self.prev_action_net(a)

        s = state['sharpe'].to(self.device)
        s = self.prev_sharpe_net(s)
        
        # Process features (from original PolicyModel)
        x = state['features'].to(self.device)
        x = torch.unsqueeze(x, 1)  # Add channel dimension
        
        # Normalization (from original PolicyModel)
        ma, idx = torch.max(x, dim=2)
        ma = ma.unsqueeze(2)
        x = x / (ma + 1e-8)  # Avoid division by zero
        
        # Feature extraction
        x = self.conv1(x)
        x = self.fc1(x)
        x = self.fc1_dropout(x)
        
        # Combine activations (from original PolicyModel)
        x = x + d + a + s
        
        # DQN output
        if self.use_dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            value = self.value_head(x)
            advantage = self.advantage_head(x)
            q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        else:
            # Standard DQN
            q_values = self.q_head(x)
        
        return q_values


class DQNValueNetwork(nn.Module):
    """
    Value network that replicates the original ValueModel architecture.
    Used for actor-critic components or as a separate value estimator.
    """
    
    def __init__(self, state_hdim: int, action_dim: int):
        super().__init__()
        
        self.state_hdim = state_hdim
        self.action_dim = action_dim
        self.hdim = 512
        self.hdim2 = 256
        
        # Replicate original ValueModel architecture exactly
        self.layer_1 = nn.Linear(state_hdim + action_dim, self.hdim)
        self.layer_1_act = nn.Softplus()
        self.layer_2 = nn.Linear(self.hdim, self.hdim2)
        self.layer_2_act = nn.Softplus()
        self.layer_3 = nn.Linear(self.hdim2, 1)
    
    def forward(self, states: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass replicating original ValueModel.
        
        Args:
            states: State features
            action: Action vector
        
        Returns:
            Value estimate
        """
        x = torch.cat((states, action), dim=1)
        
        x = self.layer_1(x)
        x = self.layer_1_act(x)
        x = self.layer_2(x) 
        x = self.layer_2_act(x)
        x = self.layer_3(x)
        
        return x


class FinancialDQNPolicyModel:
    """
    Complete financial policy model that integrates DQN with original angle_rl architecture.
    """
    
    def __init__(self,
                 data_list_file: str,
                 ticker_hash_file: str,
                 n_action_bins: int = 21,
                 dropout_ratio: float = 0.0,
                 num_conv_filters: int = 32,
                 hidden_dim: int = 47,
                 use_dueling: bool = True,
                 device: str = None):
        
        self.devmgr = get_device_manager(device)
        self.device = self.devmgr.device
        
        # Load configuration
        self.data_list_file = data_list_file
        self.ticker_hash_file = ticker_hash_file
        self.n_action_bins = n_action_bins
        
        # Load ticker hash
        with open(ticker_hash_file, 'rb') as f:
            self.ticker_hash = pickle.load(f)
        
        self.num_tickers = self.ticker_hash['num_tickers']
        self.hash_dict = self.ticker_hash['hash_D']
        
        # Action space: simplified discrete actions for DQN
        self.n_actions = min(self.num_tickers * n_action_bins, 5000)
        
        # Initialize networks
        self.policy_net = DQNPolicyNetwork(
            num_tickers=self.num_tickers,
            num_actions=self.n_actions,
            dropout_ratio=dropout_ratio,
            num_conv_filters=num_conv_filters,
            hidden_dim=hidden_dim,
            use_dueling=use_dueling,
            device=self.device
        ).to(self.device)
        
        # Value network (for auxiliary value estimation)
        self.value_net = DQNValueNetwork(
            state_hdim=hidden_dim,  # Features from policy network
            action_dim=self.num_tickers  # Portfolio allocation
        ).to(self.device)
        
        print(f"Financial DQN Policy Model initialized:")
        print(f"  Tickers: {self.num_tickers}")
        print(f"  Action space: {self.n_actions}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Device: {self.device}")
        print(f"  Dueling: {use_dueling}")
    
    def create_state_dict(self,
                         features: torch.Tensor,
                         tickers: list,
                         prev_delta: Optional[torch.Tensor] = None,
                         prev_action: Optional[torch.Tensor] = None,
                         prev_sharpe: Optional[torch.Tensor] = None) -> Dict:
        """
        Create state dictionary compatible with original PolicyModel.
        
        Args:
            features: Market features (n_stocks, feature_dim)
            tickers: List of ticker symbols
            prev_delta: Previous portfolio changes
            prev_action: Previous portfolio allocation
            prev_sharpe: Previous Sharpe ratio
        
        Returns:
            State dictionary for network input
        """
        # Map tickers to unified space
        unified_features = torch.zeros((self.num_tickers, features.shape[1]))
        
        for i, ticker in enumerate(tickers):
            if ticker in self.hash_dict and i < len(features):
                unified_idx = self.hash_dict[ticker]
                if unified_idx < self.num_tickers:
                    unified_features[unified_idx] = features[i]
        
        # Default values if not provided
        if prev_delta is None:
            prev_delta = torch.zeros(self.num_tickers)
        if prev_action is None:
            prev_action = torch.ones(self.num_tickers) / self.num_tickers
        if prev_sharpe is None:
            prev_sharpe = torch.zeros(1)
        
        state = {
            'features': unified_features,
            'delta': prev_delta,
            'action': prev_action,
            'sharpe': prev_sharpe
        }
        
        return state
    
    def select_action(self, 
                     features: torch.Tensor,
                     tickers: list,
                     prev_delta: Optional[torch.Tensor] = None,
                     prev_action: Optional[torch.Tensor] = None,
                     prev_sharpe: Optional[torch.Tensor] = None,
                     epsilon: float = 0.0) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            features: Market features
            tickers: Ticker symbols
            prev_delta: Previous portfolio changes
            prev_action: Previous portfolio allocation
            prev_sharpe: Previous Sharpe ratio
            epsilon: Exploration rate
        
        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            return np.random.randint(0, self.n_actions)
        
        # Create state
        state = self.create_state_dict(features, tickers, prev_delta, prev_action, prev_sharpe)
        
        # Forward pass
        with torch.no_grad():
            q_values = self.policy_net(state, tickers)
            action = q_values.argmax().item()
        
        return action
    
    def decode_action(self, action: int) -> torch.Tensor:
        """
        Convert discrete action to portfolio allocation (similar to original softmax output).
        
        Args:
            action: Discrete action index
        
        Returns:
            Portfolio weights tensor
        """
        portfolio = torch.zeros(self.num_tickers)
        
        if self.num_tickers <= 10:
            # Small universe: direct mapping
            stock_idx = action % self.num_tickers
            weight_idx = action // self.num_tickers
            weight = weight_idx / (self.n_action_bins - 1)
            portfolio[stock_idx] = weight
        else:
            # Large universe: focus on top stocks
            num_focus = min(20, self.num_tickers)
            focus_action = action % (num_focus * self.n_action_bins)
            stock_idx = focus_action % num_focus
            weight_idx = focus_action // num_focus
            
            # Primary weight
            primary_weight = weight_idx / (self.n_action_bins - 1)
            portfolio[stock_idx] = primary_weight
            
            # Distribute remaining
            remaining = 1.0 - primary_weight
            if remaining > 0:
                other_stocks = min(5, num_focus - 1)
                if other_stocks > 0:
                    other_weight = remaining / other_stocks
                    for i in range(other_stocks):
                        if i != stock_idx:
                            portfolio[i] = other_weight
        
        # Normalize (like softmax in original)
        total_weight = torch.sum(portfolio)
        if total_weight > 0:
            portfolio = portfolio / total_weight
        else:
            portfolio = torch.ones(self.num_tickers) / self.num_tickers
        
        return portfolio
    
    def get_q_values(self,
                    features: torch.Tensor,
                    tickers: list,
                    prev_delta: Optional[torch.Tensor] = None,
                    prev_action: Optional[torch.Tensor] = None,
                    prev_sharpe: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Get Q-values for all actions.
        
        Returns:
            Q-values tensor
        """
        state = self.create_state_dict(features, tickers, prev_delta, prev_action, prev_sharpe)
        return self.policy_net(state, tickers)
    
    def get_value_estimate(self,
                          state_features: torch.Tensor,
                          portfolio: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate using the value network.
        
        Args:
            state_features: State feature representation
            portfolio: Portfolio allocation
        
        Returns:
            Value estimate
        """
        return self.value_net(state_features, portfolio)
    
    def set_training_mode(self, training: bool = True):
        """Set networks to training or evaluation mode"""
        self.policy_net.train(training)
        self.value_net.train(training)
    
    def get_network_parameters(self):
        """Get policy network parameters for optimization"""
        return self.policy_net.parameters()
    
    def get_network_state_dict(self):
        """Get policy network state dict"""
        return self.policy_net.state_dict()
    
    def load_network_state_dict(self, state_dict):
        """Load policy network state dict"""
        self.policy_net.load_state_dict(state_dict)
    
    def to_device(self, tensor_or_model):
        """Move tensor or model to device"""
        return self.devmgr.to_dev(tensor_or_model)


def create_financial_dqn_policy(data_list_file: str,
                               ticker_hash_file: str,
                               device: str = None,
                               **kwargs) -> FinancialDQNPolicyModel:
    """
    Factory function to create financial DQN policy model.
    
    Args:
        data_list_file: Path to data list file
        ticker_hash_file: Path to ticker hash file  
        device: Device to use
        **kwargs: Additional arguments
    
    Returns:
        Initialized FinancialDQNPolicyModel
    """
    return FinancialDQNPolicyModel(
        data_list_file=data_list_file,
        ticker_hash_file=ticker_hash_file,
        device=device,
        **kwargs
    )