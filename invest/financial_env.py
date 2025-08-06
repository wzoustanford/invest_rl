"""
Financial environment that replicates the exact logic from train_RL_model.py.
This creates a proper gym-style environment that can work with DQN/R2D2.
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


class FinancialTradingEnvironment:
    """
    Environment that replicates the exact training logic from train_RL_model.py.
    Converts the actor-critic training loop into a standard RL environment interface.
    """
    
    def __init__(self,
                 data_list_filename: str,
                 ticker_hash_file: str,
                 start_date_idx: int = 0,
                 end_date_idx_plus1: int = 267,
                 action_update_interval: int = 10,
                 transaction_cost_ratio: float = 0.0015,
                 gamma: float = 0.8,
                 device: str = 'cpu'):
        
        self.data_list_filename = data_list_filename
        self.ticker_hash_file = ticker_hash_file
        self.start_date_idx = start_date_idx
        self.end_date_idx_plus1 = end_date_idx_plus1
        self.action_update_interval = action_update_interval
        self.transaction_cost_ratio = transaction_cost_ratio
        self.gamma = gamma
        self.device = torch.device(device)
        
        # Load ticker hash (exactly like train_RL_model.py)
        loadD = pickle.load(open(ticker_hash_file, 'rb'))
        self.shuffle_dict = loadD['hash_D']
        self.num_tickers = loadD['num_tickers']
        
        # Open data file handle
        self.f_data_list = open(data_list_filename, 'r')
        
        # Environment state
        self.current_step = 0
        self.current_data_idx = start_date_idx
        self.state = None
        self.prev_state = None
        
        # Episode tracking
        self.episode_ended = False
        
        print(f"Financial Trading Environment initialized:")
        print(f"  Tickers: {self.num_tickers}")
        print(f"  Data range: {start_date_idx} to {end_date_idx_plus1}")
        print(f"  Action update interval: {action_update_interval}")
        print(f"  Device: {device}")
    
    def reset(self, return_dict: bool = True) -> Tuple[Any, Dict]:
        """
        Reset environment to initial state (replicating train_RL_model.py initialization).
        
        Args:
            return_dict: If True, return state dict. If False, return flat observation.
        
        Returns:
            Initial state (dict or flat array) and info dict
        """
        # Reset file pointer
        self.f_data_list.close()
        self.f_data_list = open(self.data_list_filename, 'r')
        
        # Skip to start index
        for _ in range(self.start_date_idx):
            self.f_data_list.readline()
        
        # Reset counters
        self.current_step = 0
        self.current_data_idx = self.start_date_idx
        self.episode_ended = False
        
        # Initialize state (exactly like train_RL_model.py line 127-136)
        self.state = {
            'delta': torch.ones((1, self.num_tickers)),
            'action': 1.0 / self.num_tickers * torch.ones((1, self.num_tickers)),
            'sharpe': torch.Tensor([0.0]).view(1, 1),
            'features': None,
            'tickers': None,
            'policy_pooled_acts': torch.zeros((1, 47)),  # hidden_dim from PolicyModel
            'X': 1.0,
            'prices': torch.zeros((self.num_tickers, 1)),
        }
        
        self.prev_state = None
        
        # Load first day's data
        fl = self.f_data_list.readline().strip()
        if not fl:
            raise ValueError("No data available")
        
        D = pickle.load(open(fl, 'rb'))
        features = D['trainFeature']
        series = D['train_in_portfolio_series']
        tickers = D['all_train_tickers']
        
        # Update state with features (line 141-142)
        self.state['features'] = features
        self.state['tickers'] = tickers
        
        # Reshuffle series tensor (lines 111-124)
        shuffled_series = self._reshuffle_series(series, tickers)
        
        info = {
            'current_step': self.current_step,
            'X': self.state['X'],
            'num_tickers_available': len(tickers),
            'shuffled_series_shape': shuffled_series.shape
        }
        
        del D, series
        
        if return_dict:
            return self.state.copy(), info
        else:
            # Convert to flat observation for DQN
            flat_obs = self._state_dict_to_flat(self.state)
            return flat_obs, info
    
    def step(self, action: torch.Tensor, return_dict: bool = True) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Portfolio allocation tensor (num_tickers,)
            return_dict: If True, return state dict. If False, return flat observation.
        
        Returns:
            (next_state, reward, terminated, truncated, info)
        """
        if self.episode_ended:
            raise ValueError("Episode has ended. Call reset() first.")
        
        self.current_step += 1
        self.current_data_idx += 1
        
        # Check if episode should end
        if self.current_data_idx >= self.end_date_idx_plus1:
            self.episode_ended = True
            # Include final portfolio value in termination info
            termination_info = {
                'episode_ended': True,
                'X': float(self.state['X']),
                'final_portfolio_value': float(self.state['X'])
            }
            if return_dict:
                return self.state.copy(), 0.0, True, False, termination_info
            else:
                # Return consistent size observation
                flat_obs = self._state_dict_to_flat(self.state)
                return flat_obs, 0.0, True, False, termination_info
        
        # Load next day's data
        fl = self.f_data_list.readline().strip()
        if not fl:
            self.episode_ended = True
            if return_dict:
                return self.state.copy(), 0.0, True, False, {'no_more_data': True}
            else:
                # Return consistent size observation
                flat_obs = self._state_dict_to_flat(self.state)
                return flat_obs, 0.0, True, False, {'no_more_data': True}
        
        D = pickle.load(open(fl, 'rb'))
        features = D['trainFeature']
        series = D['train_in_portfolio_series']
        tickers = D['all_train_tickers']
        
        # Reshuffle series tensor (exactly like train_RL_model.py)
        shuffled_series = self._reshuffle_series(series, tickers)
        
        # Store previous state
        self.prev_state = self.state.copy()
        
        # Update state with new features
        self.state['features'] = features
        self.state['tickers'] = tickers
        
        # Compute returns (line 149)
        sharpe, mean_return, actual_return, stddev = self._compute_kday_returns(shuffled_series, action)
        sharpe_tensor = torch.Tensor([sharpe]).view(1, 1)
        
        # Compute delta (line 152-153)
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
        
        # Adjust sharpe with transaction costs (line 161-163)
        #sharpe_adjusted = (X - self.state['X']) / self.state['X']
        sharpe_adjusted = mean_return
        sharpe_adjusted = (sharpe_adjusted / (stddev + 1e-10))
        sharpe_tensor = torch.Tensor([sharpe_adjusted]).view(1, 1)
        
        # Reward is the sharpe ratio
        reward = float(sharpe_adjusted)
        
        # Action update logic (line 182-184)
        # Only apply new action on action_update_interval days, otherwise keep previous
        if self.current_data_idx % self.action_update_interval != 0:
            # Not an action update day - keep previous action
            actual_action = self.prev_state['action'] 
            policy_pooled_acts = self.prev_state['policy_pooled_acts']
        else:
            # Action update day - use new action
            actual_action = action
            policy_pooled_acts = torch.zeros((1, 47))  # Mock value
        
        # Create new state (lines 185-194)
        self.state = {
            'delta': delta.detach(),
            'action': actual_action.detach().view(1, -1) if hasattr(actual_action, 'detach') else actual_action.view(1, -1),
            'sharpe': sharpe_tensor.detach(),
            'policy_pooled_acts': policy_pooled_acts.detach() if hasattr(policy_pooled_acts, 'detach') else policy_pooled_acts,
            'features': None,  # Will be updated in next step
            'tickers': None,   # Will be updated in next step
            'X': X.detach() if hasattr(X, 'detach') else X,
            'prices': shuffled_series[:, 0].detach(),
        }
        
        info = {
            'sharpe': sharpe,
            'mean_return': mean_return,
            'actual_return': actual_return,
            'stddev': stddev,
            'X': float(X),
            'reward_components': {
                'sharpe_raw': sharpe,
                'sharpe_adjusted': sharpe_adjusted,
                'transaction_cost_applied': self.current_data_idx % self.action_update_interval == 0
            }
        }
        
        del D, features, series, tickers, shuffled_series
        
        if return_dict:
            return self.state.copy(), reward, False, False, info
        else:
            # Convert to flat observation for DQN
            flat_obs = self._state_dict_to_flat(self.state)
            return flat_obs, reward, False, False, info
    
    def _reshuffle_series(self, series: torch.Tensor, tickers: List[str]) -> torch.Tensor:
        """
        Reshuffle series tensor to unified hash dimensions (lines 111-124).
        """
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
    
    def _compute_kday_returns(self, shuffled_series: torch.Tensor, action_output: torch.Tensor, obj_use_mean_return: bool = True) -> Tuple[float, float, float, float]:
        """
        Compute k-day returns exactly like train_RL_model.py (lines 416-432).
        """
        # Ensure action_output is the right shape
        if action_output.dim() == 1:
            action_output = action_output.unsqueeze(1)
        
        # Portfolio shares
        portfolio_shares = action_output / torch.unsqueeze((shuffled_series[:, 0] + 1e-10), 1)
        
        # Actual return
        actual_return = torch.sum(torch.unsqueeze((shuffled_series[:, -1] - shuffled_series[:, 0]), 1) * portfolio_shares)
        
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
        """
        Compute transaction cost exactly like train_RL_model.py (lines 434-437).
        """
        # Ensure tensors are the right shape
        if current_ratios.dim() == 2:
            current_ratios = current_ratios.squeeze(0)
        if previous_ratios.dim() == 2:
            previous_ratios = previous_ratios.squeeze(0)
        
        C = torch.sum(current_X * torch.abs(current_ratios - previous_ratios) * self.transaction_cost_ratio)
        return C
    
    def get_transition_for_replay(self) -> Optional[Dict]:
        """
        Get transition data for replay buffer (lines 165-172).
        
        Returns:
            Replay buffer entry or None if not ready
        """
        if self.prev_state is None or self.current_data_idx <= self.start_date_idx:
            return None
        
        replay = {
            'prev_state': self.prev_state.copy(),
            'sharpe': self.prev_state['sharpe'],
            'state': self.state.copy(),
        }
        
        return replay
    
    def _state_dict_to_flat(self, state: Dict) -> np.ndarray:
        """
        Convert state dictionary to flat observation array for DQN.
        
        Args:
            state: State dictionary
            
        Returns:
            Flat numpy array observation
        """
        components = []
        
        # Delta (num_tickers)
        delta = state['delta'].squeeze().numpy() if isinstance(state['delta'], torch.Tensor) else state['delta']
        components.append(delta.flatten())
        
        # Previous action (num_tickers)
        action = state['action'].squeeze().numpy() if isinstance(state['action'], torch.Tensor) else state['action']
        components.append(action.flatten())
        
        # Sharpe (1)
        sharpe = state['sharpe'].squeeze().numpy() if isinstance(state['sharpe'], torch.Tensor) else state['sharpe']
        components.append(np.array([sharpe]) if np.isscalar(sharpe) else sharpe.flatten())
        
        # Portfolio value X (1)
        X = state['X']
        components.append(np.array([X]) if np.isscalar(X) else np.array(X).flatten())
        
        # Policy pooled acts (hidden_dim=47)
        acts = state['policy_pooled_acts'].squeeze().numpy() if isinstance(state['policy_pooled_acts'], torch.Tensor) else state['policy_pooled_acts']
        components.append(acts.flatten())
        
        # Features summary (if available, take mean across stocks)
        feature_dim = 249  # Fixed feature dimension
        if state['features'] is not None:
            features = state['features'].numpy() if isinstance(state['features'], torch.Tensor) else state['features']
            # Take mean across stocks for a fixed-size representation
            feature_summary = np.mean(features, axis=0)
            components.append(feature_summary.flatten())
        else:
            # Add zeros if features not available to maintain consistent size
            components.append(np.zeros(feature_dim))
        
        # Concatenate all components
        flat_obs = np.concatenate(components)
        
        # Debug: Check if dimension is wrong
        if flat_obs.shape[0] != self.get_observation_dim():
            print(f"DIMENSION MISMATCH WARNING:")
            print(f"  Expected: {self.get_observation_dim()}")
            print(f"  Actual: {flat_obs.shape[0]}")
            print(f"  Component sizes: {[c.shape[0] for c in components]}")
            # Force to correct size by truncating or padding
            expected_dim = self.get_observation_dim()
            if flat_obs.shape[0] > expected_dim:
                flat_obs = flat_obs[:expected_dim]
                print(f"  Truncated to {expected_dim}")
            elif flat_obs.shape[0] < expected_dim:
                padding = np.zeros(expected_dim - flat_obs.shape[0])
                flat_obs = np.concatenate([flat_obs, padding])
                print(f"  Padded to {expected_dim}")
        
        return flat_obs.astype(np.float32)
    
    def get_observation_dim(self) -> int:
        """Get dimension of flat observation space."""
        # Delta (num_tickers) + Action (num_tickers) + Sharpe (1) + X (1) + 
        # Policy acts (47) + Feature summary (249)
        feature_dim = 249  # From data analysis
        return 2 * self.num_tickers + 1 + 1 + 47 + feature_dim
    
    def close(self):
        """Close file handles."""
        if hasattr(self, 'f_data_list'):
            self.f_data_list.close()
    
    def __del__(self):
        """Cleanup."""
        self.close()


class FinancialEnvironmentWrapper:
    """
    Wrapper to make FinancialTradingEnvironment compatible with standard RL interfaces.
    This bridges the gap between the financial environment and DQN/R2D2 agents.
    """
    
    def __init__(self, 
                 data_list_filename: str,
                 ticker_hash_file: str,
                 use_flat_obs: bool = True,
                 discrete_actions: int = 100,
                 **kwargs):
        
        self.env = FinancialTradingEnvironment(
            data_list_filename=data_list_filename,
            ticker_hash_file=ticker_hash_file,
            **kwargs
        )
        
        self.use_flat_obs = use_flat_obs
        self.discrete_actions = discrete_actions
        
        # Action and observation spaces for compatibility
        self.num_tickers = self.env.num_tickers
        
        if use_flat_obs:
            # For DQN: discrete actions and flat observations
            self.action_space = type('ActionSpace', (), {
                'n': discrete_actions,
                'shape': (),
                'dtype': np.int64
            })()
            self.observation_space = type('ObservationSpace', (), {
                'shape': (self.env.get_observation_dim(),),
                'dtype': np.float32
            })()
        else:
            # For original: continuous actions and dict observations
            self.action_space = type('ActionSpace', (), {
                'n': self.num_tickers,
                'shape': (self.num_tickers,),
                'dtype': np.float32
            })()
            self.observation_space = type('ObservationSpace', (), {
                'shape': None,  # Variable depending on features
                'keys': ['delta', 'action', 'sharpe', 'features', 'tickers', 'policy_pooled_acts', 'X', 'prices']
            })()
        
        print(f"Financial Environment Wrapper initialized:")
        print(f"  Mode: {'DQN (flat obs)' if use_flat_obs else 'Original (dict obs)'}")
        print(f"  Action space: {f'discrete ({discrete_actions})' if use_flat_obs else f'continuous ({self.num_tickers} stocks)'}")
        print(f"  Observation space: {f'flat vector ({self.env.get_observation_dim()})' if use_flat_obs else 'state dictionary'}")
    
    def reset(self):
        """Reset environment."""
        return self.env.reset(return_dict=not self.use_flat_obs)
    
    def step(self, action):
        """Step environment."""
        # Convert action based on mode
        if self.use_flat_obs:
            # DQN mode: convert discrete to continuous
            if isinstance(action, (int, np.integer)):
                original_action = action
                action = self._discrete_to_continuous_action(action)
                # Debug logging for first few steps
                if hasattr(self, 'step_count'):
                    self.step_count += 1
                else:
                    self.step_count = 1
                
            else:
                raise ValueError(f"Expected discrete action, got {type(action)}")
        else:
            # Original mode: ensure tensor
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).float()
            elif not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.float32)
        
        return self.env.step(action, return_dict=not self.use_flat_obs)
    
    def _discrete_to_continuous_action(self, discrete_action: int) -> torch.Tensor:
        """
        Convert discrete action index to continuous portfolio allocation.
        Simple strategy for demo purposes.
        """
        portfolio = torch.zeros(self.num_tickers)
        
        # Simple mapping: focus on one stock based on action
        stock_idx = discrete_action % self.num_tickers
        weight_fraction = (discrete_action // self.num_tickers) % 21  # 21 weight levels
        weight = weight_fraction / 20.0  # 0.0 to 1.0
        
        portfolio[stock_idx] = weight
        
        # Distribute remaining weight equally among other stocks
        remaining_weight = 1.0 - weight
        if remaining_weight > 0 and self.num_tickers > 1:
            other_weight = remaining_weight / (self.num_tickers - 1)
            for i in range(self.num_tickers):
                if i != stock_idx:
                    portfolio[i] = other_weight
        
        # Normalize to ensure sum = 1
        total = torch.sum(portfolio)
        if total > 0:
            portfolio = portfolio / total
        else:
            portfolio = torch.ones(self.num_tickers) / self.num_tickers
        
        return portfolio
    
    def get_replay_transition(self):
        """Get transition for replay buffer."""
        return self.env.get_transition_for_replay()
    
    def close(self):
        """Close environment."""
        self.env.close()


# Factory functions for easy creation
def create_financial_environment(data_list_filename: str,
                                ticker_hash_file: str,
                                **kwargs) -> FinancialEnvironmentWrapper:
    """
    Create financial environment wrapper.
    
    Args:
        data_list_filename: Path to data list file
        ticker_hash_file: Path to ticker hash file
        **kwargs: Additional environment parameters
    
    Returns:
        FinancialEnvironmentWrapper instance
    """
    return FinancialEnvironmentWrapper(
        data_list_filename=data_list_filename,
        ticker_hash_file=ticker_hash_file,
        **kwargs
    )
