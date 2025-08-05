"""
Financial environment wrapper to make financial data compatible with DQN/R2D2 from angle/RL repo.
This allows us to use the game RL algorithms on financial time series data.
"""

import torch
import numpy as np
import pickle
import sys
import os
from typing import Dict, List, Optional, Tuple
from collections import deque

# Add the angle/RL repo to path so we can import from it
sys.path.append('/home/ubuntu/code/angle/RL')

# Import with fallback
try:
    from model.device_utils import get_device_manager
except ImportError:
    # Fallback implementation
    class MockDeviceManager:
        def __init__(self, device=None):
            import torch
            self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        def to_dev(self, x):
            return x.to(self.device) if hasattr(x, 'to') else x
    
    def get_device_manager(device=None):
        return MockDeviceManager(device)


class FinancialEnvironment:
    """
    Financial trading environment that adapts financial data for DQN/R2D2 training.
    
    Observations: Market features + portfolio state  
    Actions: Portfolio allocation (discrete bins for each stock)
    Rewards: Sharpe ratio (risk-adjusted returns)
    """
    
    def __init__(self, 
                 data_list_file: str,
                 ticker_hash_file: str,
                 num_action_bins: int = 21,  # 0%, 5%, 10%, ..., 100% allocations
                 transaction_cost: float = 0.0015,
                 action_update_interval: int = 10,
                 max_episode_steps: int = 200,
                 feature_aggregation: str = 'mean',  # 'mean', 'attention', 'cnn'
                 device: str = None):
        
        self.devmgr = get_device_manager(device)
        self.device = self.devmgr.device
        
        # Load financial data
        self.data_list_file = data_list_file
        self.ticker_hash_file = ticker_hash_file
        self.data_files = self._load_data_list()
        self.ticker_hash = self._load_ticker_hash()
        
        # Environment parameters
        self.num_action_bins = num_action_bins
        self.transaction_cost = transaction_cost
        self.action_update_interval = action_update_interval
        self.max_episode_steps = max_episode_steps
        self.feature_aggregation = feature_aggregation
        
        # State tracking
        self.current_step = 0
        self.current_data_idx = 0
        self.portfolio_value = 1.0
        self.previous_portfolio = None
        self.previous_sharpe = 0.0
        self.episode_returns = []
        
        # Environment properties
        self.num_stocks = self.ticker_hash['num_tickers']
        self.feature_dim = 249  # From data analysis
        
        # Create simplified observation space for DQN
        if feature_aggregation == 'mean':
            # Aggregate to fixed size: features + portfolio + scalars
            obs_dim = self.feature_dim + min(self.num_stocks, 100) + 5
        elif feature_aggregation == 'cnn':
            # Use CNN-like structure (channels, height, width)
            obs_dim = (4, 84, 84)  # Compatible with Atari DQN
        else:
            obs_dim = self.feature_dim + 10  # Simplified
        
        self.obs_dim = obs_dim
        
        # Action space: single discrete action (we'll map internally)
        # For simplicity, use smaller action space
        self.n_actions = min(self.num_stocks * num_action_bins, 10000)  # Cap for feasibility
        
        print(f"Financial Environment initialized:")
        print(f"  Observation dim: {obs_dim}")
        print(f"  Action space: {self.n_actions} discrete actions")
        print(f"  Stocks: {self.num_stocks}")
        print(f"  Device: {self.device}")
    
    def _load_data_list(self) -> List[str]:
        """Load list of data file paths"""
        with open(self.data_list_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    def _load_ticker_hash(self) -> Dict:
        """Load ticker hash mapping"""
        with open(self.ticker_hash_file, 'rb') as f:
            return pickle.load(f)
    
    def _load_daily_data(self, data_idx: int) -> Optional[Dict]:
        """Load data for a specific day"""
        if data_idx >= len(self.data_files):
            return None
            
        try:
            with open(self.data_files[data_idx], 'rb') as f:
                data = pickle.load(f)
            
            return {
                'features': data['trainFeature'],
                'prices': data['train_in_portfolio_series'],
                'tickers': data['all_train_tickers']
            }
        except Exception as e:
            print(f"Error loading data {data_idx}: {e}")
            return None
    
    def _create_observation(self, daily_data: Dict) -> np.ndarray:
        """Create observation compatible with DQN"""
        if daily_data is None:
            if self.feature_aggregation == 'cnn':
                return np.zeros((4, 84, 84), dtype=np.float32)
            else:
                return np.zeros(self.obs_dim, dtype=np.float32)
        
        features = daily_data['features']
        tickers = daily_data['tickers']
        
        if self.feature_aggregation == 'mean':
            return self._create_mean_observation(features, tickers)
        elif self.feature_aggregation == 'cnn':
            return self._create_cnn_observation(features, tickers)
        else:
            return self._create_simple_observation(features, tickers)
    
    def _create_mean_observation(self, features: torch.Tensor, tickers: List[str]) -> np.ndarray:
        """Create mean-aggregated observation"""
        # Map tickers to unified space
        unified_features = torch.zeros((self.num_stocks, self.feature_dim))
        portfolio_state = torch.zeros(min(self.num_stocks, 100))
        
        hash_dict = self.ticker_hash['hash_D']
        valid_count = 0
        
        for i, ticker in enumerate(tickers):
            if ticker in hash_dict and i < len(features):
                unified_idx = hash_dict[ticker]
                if unified_idx < self.num_stocks:
                    unified_features[unified_idx] = features[i]
                    valid_count += 1
                    if unified_idx < 100:  # Portfolio state (top 100 stocks)
                        if self.previous_portfolio is not None and unified_idx < len(self.previous_portfolio):
                            portfolio_state[unified_idx] = self.previous_portfolio[unified_idx]
        
        # Aggregate features (mean of valid stocks)
        if valid_count > 0:
            aggregated_features = torch.mean(unified_features[:valid_count], dim=0)
        else:
            aggregated_features = torch.zeros(self.feature_dim)
        
        # Create observation
        obs = torch.cat([
            aggregated_features,
            portfolio_state,
            torch.tensor([
                self.portfolio_value,
                self.previous_sharpe,
                self.current_step / self.max_episode_steps,
                valid_count / max(1, len(tickers)),
                len(self.episode_returns) / 100.0  # Episode progress
            ])
        ])
        
        return obs.numpy().astype(np.float32)
    
    def _create_cnn_observation(self, features: torch.Tensor, tickers: List[str]) -> np.ndarray:
        """Create CNN-compatible observation (like Atari frames)"""
        # Create 4-channel 84x84 image from financial data
        obs = np.zeros((4, 84, 84), dtype=np.float32)
        
        # Channel 0: Price changes (normalized)
        if len(features) > 0:
            price_features = features[:, :84].T  # Take first 84 features as "price"
            if price_features.shape[0] < 84:
                price_features = torch.nn.functional.pad(price_features, (0, 0, 0, 84 - price_features.shape[0]))
            obs[0] = price_features[:84, :84].numpy()
        
        # Channel 1: Volume/momentum features
        if len(features) > 84:
            volume_features = features[:, 84:168].T
            if volume_features.shape[0] < 84:
                volume_features = torch.nn.functional.pad(volume_features, (0, 0, 0, 84 - volume_features.shape[0]))
            obs[1] = volume_features[:84, :84].numpy()
        
        # Channel 2: Technical indicators
        if len(features) > 168:
            tech_features = features[:, 168:].T
            if tech_features.shape[0] < 84:
                tech_features = torch.nn.functional.pad(tech_features, (0, 0, 0, 84 - tech_features.shape[0]))
            obs[2] = tech_features[:84, :84].numpy()
        
        # Channel 3: Portfolio state
        if self.previous_portfolio is not None:
            portfolio_grid = torch.zeros(84, 84)
            # Fill grid with portfolio weights (repeated pattern)
            portfolio_subset = self.previous_portfolio[:min(len(self.previous_portfolio), 84*84)]
            for i, weight in enumerate(portfolio_subset):
                row, col = i // 84, i % 84
                if row < 84:
                    portfolio_grid[row, col] = weight
            obs[3] = portfolio_grid.numpy()
        
        # Normalize to [0, 1]
        obs = np.clip(obs, -5, 5) / 10.0 + 0.5
        
        return obs
    
    def _create_simple_observation(self, features: torch.Tensor, tickers: List[str]) -> np.ndarray:
        """Create simple observation"""
        # Just take mean of features and add portfolio state
        if len(features) > 0:
            mean_features = torch.mean(features, dim=0)[:self.feature_dim]
        else:
            mean_features = torch.zeros(self.feature_dim)
        
        obs = torch.cat([
            mean_features,
            torch.tensor([
                self.portfolio_value,
                self.previous_sharpe,
                self.current_step / self.max_episode_steps,
                len(tickers) / 1000.0,  # Number of stocks
                np.mean(self.episode_returns) if self.episode_returns else 0.0,
                np.std(self.episode_returns) if len(self.episode_returns) > 1 else 0.0,
                len(self.episode_returns) / 100.0,
                0.0, 0.0, 0.0  # Padding
            ])
        ])
        
        return obs.numpy().astype(np.float32)
    
    def _decode_action(self, action: int) -> torch.Tensor:
        """Convert single discrete action to portfolio weights"""
        # Simple strategy: map action to focus on specific stocks
        portfolio = torch.zeros(self.num_stocks)
        
        if self.num_stocks <= 10:
            # Small universe: direct mapping
            stock_idx = action % self.num_stocks
            weight_idx = action // self.num_stocks
            weight = weight_idx / (self.num_action_bins - 1)
            portfolio[stock_idx] = weight
        else:
            # Large universe: focus on top stocks with action pattern
            num_focus_stocks = min(20, self.num_stocks)  # Focus on top 20 stocks
            
            # Distribute action across focus stocks
            focus_action = action % (num_focus_stocks * self.num_action_bins)
            stock_idx = focus_action % num_focus_stocks
            weight_idx = focus_action // num_focus_stocks
            
            # Set primary allocation
            primary_weight = weight_idx / (self.num_action_bins - 1)
            portfolio[stock_idx] = primary_weight
            
            # Add small allocations to other top stocks
            remaining_weight = 1.0 - primary_weight
            if remaining_weight > 0:
                other_stocks = min(5, num_focus_stocks - 1)
                if other_stocks > 0:
                    other_weight = remaining_weight / other_stocks
                    for i in range(other_stocks):
                        if i != stock_idx:
                            portfolio[i] = other_weight
        
        # Normalize to sum to 1
        total_weight = torch.sum(portfolio)
        if total_weight > 0:
            portfolio = portfolio / total_weight
        else:
            portfolio = torch.ones(self.num_stocks) / self.num_stocks
        
        return portfolio
    
    def _compute_returns(self, daily_data: Dict, portfolio: torch.Tensor) -> Tuple[float, float, float, float]:
        """Compute portfolio returns"""
        if daily_data is None:
            return 0.0, 0.0, 0.0, 1.0
        
        prices = daily_data['prices']
        tickers = daily_data['tickers']
        
        if prices.shape[1] < 2:
            return 0.0, 0.0, 0.0, 1.0
        
        # Map to unified space
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
        actual_return = torch.sum((final_prices - initial_prices) * portfolio_shares)
        
        # Time series returns for Sharpe calculation
        if unified_prices.shape[1] > 1:
            returns_series = []
            for t in range(1, unified_prices.shape[1]):
                period_return = torch.sum((unified_prices[:, t] - unified_prices[:, t-1]) * portfolio_shares)
                returns_series.append(period_return.item())
            
            if len(returns_series) > 1:
                mean_return = np.mean(returns_series)
                stddev = np.std(returns_series) + 1e-8
                sharpe = mean_return / stddev
                
                self.episode_returns.extend(returns_series)
                return sharpe, mean_return, actual_return.item(), stddev
        
        return 0.0, 0.0, actual_return.item(), 1.0
    
    def reset(self) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        # Reset state
        self.current_step = 0
        start_idx = np.random.randint(0, max(1, len(self.data_files) - self.max_episode_steps - 50))
        self.current_data_idx = start_idx
        self.portfolio_value = 1.0
        self.previous_portfolio = torch.ones(self.num_stocks) / self.num_stocks
        self.previous_sharpe = 0.0
        self.episode_returns = []
        
        # Load first observation
        daily_data = self._load_daily_data(self.current_data_idx)
        observation = self._create_observation(daily_data)
        
        info = {'portfolio_value': self.portfolio_value}
        return observation, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute environment step"""
        # Load current day data
        daily_data = self._load_daily_data(self.current_data_idx)
        if daily_data is None:
            # Episode finished
            obs = np.zeros_like(self._create_observation(None))
            return obs, 0.0, True, False, {'portfolio_value': self.portfolio_value}
        
        # Decode action to portfolio
        new_portfolio = self._decode_action(action)
        
        # Compute returns
        sharpe, mean_return, actual_return, stddev = self._compute_returns(daily_data, new_portfolio)
        
        # Update portfolio value
        if self.current_step % self.action_update_interval == 0:
            # Rebalance with transaction costs
            if self.previous_portfolio is not None:
                transaction_cost = torch.sum(torch.abs(new_portfolio - self.previous_portfolio)) * self.transaction_cost
                self.portfolio_value *= (1 - transaction_cost)
            self.previous_portfolio = new_portfolio.clone()
        
        # Apply returns
        self.portfolio_value *= (1 + actual_return / 100)
        self.previous_sharpe = sharpe
        
        # Reward
        reward = float(sharpe)
        
        # Prevent extreme losses
        if self.portfolio_value < 0.5:
            reward -= 10.0  # Large negative reward for major losses
        elif self.portfolio_value > 2.0:
            reward += 1.0   # Bonus for good performance
        
        # Update state
        self.current_step += 1
        self.current_data_idx += 1
        
        # Check termination
        terminated = (self.current_step >= self.max_episode_steps or 
                     self.current_data_idx >= len(self.data_files) or
                     self.portfolio_value <= 0.3)
        
        # Next observation
        if not terminated:
            next_daily_data = self._load_daily_data(self.current_data_idx)
            observation = self._create_observation(next_daily_data)
        else:
            observation = self._create_observation(None)
        
        info = {
            'portfolio_value': self.portfolio_value,
            'sharpe': sharpe,
            'mean_return': mean_return,
            'actual_return': actual_return,
            'step': self.current_step
        }
        
        return observation, reward, terminated, False, info


class FinancialDQNWrapper:
    """Wrapper to make financial environment compatible with angle/RL DQN"""
    
    def __init__(self, 
                 data_list_file: str,
                 ticker_hash_file: str,
                 feature_aggregation: str = 'mean',
                 max_episode_steps: int = 100,
                 device: str = None):
        
        self.env = FinancialEnvironment(
            data_list_file=data_list_file,
            ticker_hash_file=ticker_hash_file,
            feature_aggregation=feature_aggregation,
            max_episode_steps=max_episode_steps,
            device=device
        )
        
        # Properties for DQN compatibility
        self.action_space = type('ActionSpace', (), {'n': self.env.n_actions})()
        self.observation_space = self.env.obs_dim
    
    def reset(self):
        """Reset for DQN compatibility"""
        obs, info = self.env.reset()
        return obs, info
    
    def step(self, action):
        """Step for DQN compatibility"""
        return self.env.step(action)
    
    def close(self):
        """Close environment"""
        pass