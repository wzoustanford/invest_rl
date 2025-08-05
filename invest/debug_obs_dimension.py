#!/usr/bin/env python3
"""
Debug observation dimension mismatch
"""

import sys
import torch
import numpy as np

sys.path.append('/home/ubuntu/code/angle_rl/invest')

from financial_env import create_financial_environment

def debug_observation_dimensions():
    """Debug the observation dimension issue."""
    
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/dqn_large_exp_ticker_hash.pkl"
    
    # Create environment with same settings as training
    env = create_financial_environment(
        data_list_filename=data_list_file,
        ticker_hash_file=ticker_hash_file,
        use_flat_obs=True,  # DQN mode
        discrete_actions=200,
        start_date_idx=0,
        end_date_idx_plus1=265,
        action_update_interval=10,  # 10-day trading
        transaction_cost_ratio=0.0015,
        device='cuda'
    )
    
    print("=== Observation Dimension Debug ===")
    
    # Reset and check initial observation
    obs, info = env.reset()
    print(f"Environment observation_space: {env.observation_space}")
    print(f"Reset observation shape: {obs.shape}")
    print(f"Reset observation type: {type(obs)}")
    
    # Take a step and check observation
    action = 50  # Random action
    next_obs, reward, terminated, truncated, step_info = env.step(action)
    print(f"Step observation shape: {next_obs.shape}")
    print(f"Step observation type: {type(next_obs)}")
    
    # Take multiple steps to see if dimension changes
    for i in range(5):
        action = np.random.randint(0, 200)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        print(f"Step {i+2} observation shape: {next_obs.shape}")
        
        if terminated or truncated:
            print(f"Episode ended at step {i+2}")
            break
    
    env.close()

if __name__ == "__main__":
    debug_observation_dimensions()