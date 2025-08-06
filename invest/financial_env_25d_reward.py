"""
Financial environment with MODIFIED REWARD FUNCTION.
This uses 25-day series return instead of step-by-step portfolio value changes.

Key change: reward = actual_return / stddev instead of (X_new - X_old)/X_old / stddev
"""

import torch
import numpy as np
import pickle
import sys
import os
from typing import Dict, List, Optional, Tuple, Any
from collections import deque

# Import utility functions
import utils


class FinancialTradingEnvironment25D:
    """
    Environment with MODIFIED REWARD FUNCTION for 25-day series returns.
    
    Original reward: (X_new - X_old)/X_old / stddev  (immediate portfolio changes)
    New reward: actual_return / stddev  (25-day series return)
    """
    
    def __init__(self,
                 data_list_filename: str,
                 ticker_hash_file: str,
                 start_date_idx: int = 0,
                 end_date_idx_plus1: int = 267,
                 action_update_interval: int = 10,
                 transaction_cost_ratio: float = 0.0015,
                 device: str = 'cpu',
                 news_features: bool = False):
        
        self.data_list_filename = data_list_filename
        self.ticker_hash_file = ticker_hash_file
        self.start_date_idx = start_date_idx
        self.end_date_idx_plus1 = end_date_idx_plus1
        self.action_update_interval = action_update_interval
        self.transaction_cost_ratio = transaction_cost_ratio
        self.device = device
        self.news_features = news_features
        
        # Load data
        self._load_data()
        
        # Initialize environment state
        self.reset()
        
        print(f"Financial Trading Environment initialized:")
        print(f"  Tickers: {self.num_tickers}")
        print(f"  Data range: {start_date_idx} to {end_date_idx_plus1}")
        print(f"  Action update interval: {action_update_interval}")
        print(f"  Device: {device}")
        print(f"  🎯 REWARD: 25-day series return (modified)")
    
    def _load_data(self):
        """Load ticker hash and data files."""
        # Load ticker hash
        with open(self.ticker_hash_file, 'rb') as f:
            self.ticker_data = pickle.load(f)
        
        # Extract tickers from hash_D (ticker -> index mapping)
        if 'tickers' in self.ticker_data:
            self.tickers = self.ticker_data['tickers']
            self.num_tickers = len(self.tickers)
        elif 'hash_D' in self.ticker_data:
            # Create ticker list from hash_D mapping
            hash_d = self.ticker_data['hash_D']
            self.tickers = [''] * len(hash_d)
            for ticker, idx in hash_d.items():
                self.tickers[idx] = ticker
            self.num_tickers = len(self.tickers)
        else:
            raise ValueError("Ticker hash file must contain 'tickers' or 'hash_D' key")
        
        # Create shuffle dict for reshuffling series
        self.shuffle_dict = self.ticker_data['hash_D']
        
        # Load data file list
        with open(self.data_list_filename, 'r') as f:
            self.data_files = [line.strip() for line in f.readlines() if line.strip()]
    
    def reset(self) -> Dict:
        """Reset environment to initial state."""
        self.current_data_idx = self.start_date_idx
        self.state = None
        self.prev_state = None
        self.done = False
        
        # Initialize first observation
        return self._get_observation()
    
    def _get_observation(self) -> Dict:
        """Get current observation state."""
        if self.current_data_idx >= self.end_date_idx_plus1:
            self.done = True
            return self.state if self.state else {}
        
        # Load current data file (line 107-109)
        current_file = self.data_files[self.current_data_idx]
        with open(current_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract state information from data file
        features = data['trainFeature']
        series = data['train_in_portfolio_series']
        tickers = data['all_train_tickers']
        
        # Reshuffle series tensor to match ticker order
        shuffled_series = self._reshuffle_series(series, tickers)
        
        # Convert to tensors and move to device
        shuffled_series = shuffled_series.to(self.device)
        features = features.to(self.device)
        
        # Initialize state for the first time
        if self.state is None:
            # Initial action (uniform allocation)
            initial_action = torch.ones(self.num_tickers, device=self.device) / self.num_tickers
            initial_action = initial_action.view(1, -1)
            
            # Initial values
            delta = torch.ones((1, self.num_tickers), device=self.device)
            X = torch.tensor(1.0, device=self.device)  # Initial portfolio value
            sharpe_tensor = torch.zeros((1, 1), device=self.device)
            policy_pooled_acts = torch.zeros((1, 47), device=self.device)
            
            self.state = {
                'delta': delta,
                'action': initial_action,
                'sharpe': sharpe_tensor,
                'policy_pooled_acts': policy_pooled_acts,
                'features': features,
                'tickers': tickers,
                'X': X,
                'prices': shuffled_series[:, 0].detach(),
            }
        else:
            # Update features and tickers for current timestep
            self.state['features'] = features
            self.state['tickers'] = tickers
        
        return self.state
    
    def step(self, action: torch.Tensor) -> Tuple[Dict, float, bool, Dict]:
        """
        Take environment step with MODIFIED REWARD FUNCTION.
        
        Key change: Uses 25-day series return instead of portfolio value changes.
        """
        if self.done:
            return self.state, 0.0, True, {}
        
        # Store previous state
        self.prev_state = {k: v.clone() if hasattr(v, 'clone') else v for k, v in self.state.items()}
        
        # Load current data
        current_file = self.data_files[self.current_data_idx]
        with open(current_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract state information from data file
        features = data['trainFeature']
        series = data['train_in_portfolio_series']
        tickers = data['all_train_tickers']
        
        # Reshuffle series tensor to match ticker order
        shuffled_series = self._reshuffle_series(series, tickers)
        shuffled_series = shuffled_series.to(self.device)
        
        # Compute k-day returns (lines 134-137)
        sharpe, mean_return, actual_return, stddev = self._compute_kday_returns(
            shuffled_series, self.state['action']
        )
        
        # Compute price changes delta (line 151-153)
        if self.current_data_idx == self.start_date_idx + 1:
            delta = torch.ones((1, self.num_tickers))
        else:
            delta = shuffled_series[:, 0] / (self.state['prices'] + 1e-10)
            delta = delta.view(1, -1)
        
        # Compute portfolio value X (line 154)
        X = torch.sum(delta * self.state['action'] * self.state['X'])
        
        # Apply transaction costs if needed (line 157-158)
        if self.current_data_idx % self.action_update_interval == 0:
            X = X - self._transaction_cost(action, self.state['action'], X)
        
        # 🎯 MODIFIED REWARD FUNCTION: Use 25-day series return instead of portfolio changes
        # Original: reward = (X - self.state['X']) / self.state['X'] / stddev
        # New: reward = actual_return / stddev (25-day return)
        reward = float(actual_return / (stddev + 1e-10))
        
        # Update sharpe tensor (keeping this for consistency with state structure)
        sharpe_adjusted = (X - self.state['X']) / self.state['X']
        sharpe_tensor = torch.Tensor([sharpe_adjusted]).view(1, 1)
        
        # Action update logic (line 182-184)
        if self.current_data_idx % self.action_update_interval != 0:
            # Not an action update day - keep previous action
            actual_action = self.prev_state['action'] 
            policy_pooled_acts = self.prev_state['policy_pooled_acts']
        else:
            # Action update day - use new action
            actual_action = action
            policy_pooled_acts = torch.zeros((1, 47))
        
        # Create new state
        self.state = {
            'delta': delta.detach(),
            'action': actual_action.detach().view(1, -1) if hasattr(actual_action, 'detach') else actual_action.view(1, -1),
            'sharpe': sharpe_tensor.detach(),
            'policy_pooled_acts': policy_pooled_acts.detach() if hasattr(policy_pooled_acts, 'detach') else policy_pooled_acts,
            'features': features.detach(),
            'tickers': tickers,
            'X': X.detach() if hasattr(X, 'detach') else X,
            'prices': shuffled_series[:, 0].detach(),
        }
        
        # Move to next timestep
        self.current_data_idx += 1
        
        # Check if done
        if self.current_data_idx >= self.end_date_idx_plus1:
            self.done = True
        
        info = {
            'sharpe': sharpe,
            'mean_return': mean_return,
            'actual_return': actual_return,
            'stddev': stddev,
            'portfolio_value': float(X),
            'reward_type': '25d_series_return',  # Mark this reward type
            'original_portfolio_reward': float((X - self.prev_state['X']) / self.prev_state['X'] / (stddev + 1e-10))  # For comparison
        }
        
        return self.state, reward, self.done, info
    
    def _reshuffle_series(self, series: torch.Tensor, tickers: List[str]) -> torch.Tensor:
        """Reshuffle series tensor to unified hash dimensions."""
        indices = []
        mask = []
        for t in tickers:
            if t in self.shuffle_dict:
                indices.append(self.shuffle_dict[t])
                mask.append(True)
            else:
                mask.append(False)
        
        indices = torch.Tensor(indices).to(int)
        
        base_frame = torch.zeros((self.num_tickers, series.shape[1]))
        base_frame[indices] = series[mask]
        
        return base_frame
    
    def _compute_kday_returns(self, shuffled_series: torch.Tensor, action_output: torch.Tensor, 
                            obj_use_mean_return: bool = False) -> Tuple[float, float, float, float]:
        """
        Compute k-day returns exactly like train_RL_model.py (lines 134-137).
        """
        # Ensure action_output is the right shape
        if action_output.dim() == 1:
            action_output = action_output.unsqueeze(1)
        
        # Portfolio shares
        portfolio_shares = action_output / torch.unsqueeze((shuffled_series[:, 0] + 1e-10), 1)
        
        # 🎯 This is the 25-day return calculation we're now using for rewards
        # Fixed: Add epsilon to prevent division by zero
        actual_return = torch.sum(torch.unsqueeze((shuffled_series[:, -1] - shuffled_series[:, 0]) / (shuffled_series[:, 0] + 1e-10), 1) * portfolio_shares)
        
        # Returns series
        returns_series = torch.sum(shuffled_series[:, 1:] * portfolio_shares - torch.unsqueeze(shuffled_series[:, 0], 1) * portfolio_shares, dim=0)
        
        mean_return = torch.mean(returns_series)
        stddev = torch.std(returns_series)
        
        if obj_use_mean_return:
            sharpe = mean_return / (stddev + 1e-10)
        else:
            sharpe = actual_return / (stddev + 1e-10)
        
        return sharpe.item(), mean_return.item(), actual_return.item(), stddev.item()
    
    def _transaction_cost(self, current_ratios: torch.Tensor, previous_ratios: torch.Tensor, current_X: torch.Tensor) -> torch.Tensor:
        """Compute transaction cost exactly like train_RL_model.py (lines 434-437)."""
        # Ensure tensors are the right shape
        if current_ratios.dim() == 2:
            current_ratios = current_ratios.squeeze(0)
        if previous_ratios.dim() == 2:
            previous_ratios = previous_ratios.squeeze(0)
        
        C = torch.sum(current_X * torch.abs(current_ratios - previous_ratios) * self.transaction_cost_ratio)
        return C
    
    def get_transition_for_replay(self) -> Optional[Dict]:
        """Get transition data for replay buffer."""
        if self.prev_state is None or self.current_data_idx <= self.start_date_idx:
            return None
        
        replay = {
            'prev_state': self.prev_state.copy(),
            'sharpe': self.prev_state['sharpe'],
        }
        return replay


class FinancialEnvironmentWrapper25D:
    """
    Wrapper to convert FinancialTradingEnvironment25D to standard RL interface.
    """
    
    def __init__(self,
                 data_list_filename: str,
                 ticker_hash_file: str,
                 start_date_idx: int = 0,
                 end_date_idx_plus1: int = 267,
                 action_update_interval: int = 10,
                 transaction_cost_ratio: float = 0.0015,
                 device: str = 'cpu',
                 mode: str = 'dqn',
                 num_discrete_actions: int = 200):
        
        self.env = FinancialTradingEnvironment25D(
            data_list_filename=data_list_filename,
            ticker_hash_file=ticker_hash_file,
            start_date_idx=start_date_idx,
            end_date_idx_plus1=end_date_idx_plus1,
            action_update_interval=action_update_interval,
            transaction_cost_ratio=transaction_cost_ratio,
            device=device
        )
        
        self.mode = mode
        self.num_discrete_actions = num_discrete_actions
        self.device = device
        
        # Calculate observation dimensions
        self.num_tickers = self.env.num_tickers
        
        if mode == 'dqn':
            # Flat observation for DQN
            self.observation_dim = (self.num_tickers * 2 + 1 + 1 + 47 + 251,)  # Features + context
            self.action_space_size = num_discrete_actions
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        print(f"Financial Environment Wrapper initialized:")
        print(f"  Mode: {mode.upper()} (flat obs)")
        print(f"  Action space: discrete ({num_discrete_actions})")
        print(f"  Observation space: flat vector ({self.observation_dim[0]})")
        print(f"  🎯 REWARD: 25-day series return")
    
    def reset(self):
        """Reset environment."""
        state = self.env.reset()
        return self._process_observation(state)
    
    def step(self, action):
        """Take environment step."""
        # Convert discrete action to continuous action allocation
        action_tensor = self._convert_action(action)
        
        # Take step
        next_state, reward, done, info = self.env.step(action_tensor)
        
        # Process observation
        next_obs = self._process_observation(next_state)
        
        return next_obs, reward, done, info
    
    def _convert_action(self, action):
        """Convert discrete action to continuous portfolio allocation."""
        if self.mode == 'dqn':
            # Convert single discrete action to allocation vector
            action_idx = action.item() if hasattr(action, 'item') else action
            
            # Simple strategy: focus allocation on different ticker groups
            allocation = torch.zeros(self.num_tickers, device=self.device)
            
            # Map discrete action to portfolio allocation pattern
            tickers_per_action = max(1, self.num_tickers // self.num_discrete_actions)
            start_idx = (action_idx * tickers_per_action) % self.num_tickers
            end_idx = min(start_idx + tickers_per_action, self.num_tickers)
            
            # Allocate equally among selected tickers
            selected_count = end_idx - start_idx
            if selected_count > 0:
                allocation[start_idx:end_idx] = 1.0 / selected_count
            else:
                allocation[0] = 1.0  # Fallback to first ticker
            
            return allocation.view(1, -1)
        
        raise ValueError(f"Unsupported mode: {self.mode}")
    
    def _process_observation(self, state):
        """Convert environment state to observation."""
        if not state:
            return torch.zeros(self.observation_dim, device=self.device)
        
        if self.mode == 'dqn':
            # Create flat observation vector
            components = []
            
            # Previous action allocation (num_tickers)
            if 'action' in state:
                action_flat = state['action'].view(-1)
                if len(action_flat) != self.num_tickers:
                    print(f"DIMENSION WARNING: Expected action dim {self.num_tickers}, got {len(action_flat)}")
                    if len(action_flat) > self.num_tickers:
                        action_flat = action_flat[:self.num_tickers]
                    else:
                        action_flat = torch.cat([action_flat, torch.zeros(self.num_tickers - len(action_flat), device=self.device)])
                components.append(action_flat)
            else:
                components.append(torch.zeros(self.num_tickers, device=self.device))
            
            # Current prices/delta (num_tickers)
            if 'delta' in state:
                delta_flat = state['delta'].view(-1)
                if len(delta_flat) != self.num_tickers:
                    if len(delta_flat) > self.num_tickers:
                        delta_flat = delta_flat[:self.num_tickers]
                    else:
                        delta_flat = torch.cat([delta_flat, torch.zeros(self.num_tickers - len(delta_flat), device=self.device)])
                components.append(delta_flat)
            else:
                components.append(torch.zeros(self.num_tickers, device=self.device))
            
            # Portfolio value (1)
            if 'X' in state:
                components.append(state['X'].view(1))
            else:
                components.append(torch.ones(1, device=self.device))
            
            # Sharpe ratio (1)
            if 'sharpe' in state:
                components.append(state['sharpe'].view(-1))
            else:
                components.append(torch.zeros(1, device=self.device))
            
            # Policy pooled acts (47)
            if 'policy_pooled_acts' in state:
                pooled_flat = state['policy_pooled_acts'].view(-1)
                if len(pooled_flat) != 47:
                    if len(pooled_flat) > 47:
                        pooled_flat = pooled_flat[:47]
                    else:
                        pooled_flat = torch.cat([pooled_flat, torch.zeros(47 - len(pooled_flat), device=self.device)])
                components.append(pooled_flat)
            else:
                components.append(torch.zeros(47, device=self.device))
            
            # Features (variable, pad/truncate to 251)
            if 'features' in state and state['features'] is not None:
                features_flat = state['features'].view(-1)
                target_feature_dim = 251
                if len(features_flat) != target_feature_dim:
                    if len(features_flat) > target_feature_dim:
                        features_flat = features_flat[:target_feature_dim]
                        print(f"DIMENSION MISMATCH WARNING:")
                        print(f"  Expected: {self.observation_dim[0]}")
                        print(f"  Actual: {sum(len(c) for c in components[:-1]) + len(features_flat)}")
                        print(f"  Component sizes: {[len(c) for c in components[:-1]] + [len(features_flat)]}")
                        print(f"  Truncated to {self.observation_dim[0]}")
                    else:
                        features_flat = torch.cat([features_flat, torch.zeros(target_feature_dim - len(features_flat), device=self.device)])
                        print(f"DIMENSION MISMATCH WARNING:")
                        print(f"  Expected: {self.observation_dim[0]}")
                        print(f"  Actual: {sum(len(c) for c in components[:-1]) + len(features_flat)}")
                        print(f"  Component sizes: {[len(c) for c in components[:-1]] + [len(features_flat)]}")
                        print(f"  Padded to {self.observation_dim[0]}")
                components.append(features_flat)
            else:
                components.append(torch.zeros(251, device=self.device))
            
            # Concatenate all components
            obs = torch.cat(components)
            
            # Ensure correct final dimension
            if len(obs) != self.observation_dim[0]:
                if len(obs) > self.observation_dim[0]:
                    obs = obs[:self.observation_dim[0]]
                else:
                    obs = torch.cat([obs, torch.zeros(self.observation_dim[0] - len(obs), device=self.device)])
            
            return obs
        
        raise ValueError(f"Unsupported mode: {self.mode}")
    
    @property
    def observation_space(self):
        """Get observation space."""
        return self.observation_dim
    
    @property
    def action_space(self):
        """Get action space."""
        return self.action_space_size