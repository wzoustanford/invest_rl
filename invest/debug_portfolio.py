#!/usr/bin/env python3
"""
Quick test to debug portfolio value issue
"""

import sys
import torch
import numpy as np

sys.path.append('/home/ubuntu/code/angle_rl/invest')

from financial_env import create_financial_environment
from financial_dqn_agent import create_financial_dqn_agent

def test_portfolio_debug():
    """Test portfolio value calculations with debugging."""
    
    # Use existing ticker hash from previous run
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/dqn_large_exp_ticker_hash.pkl"
    
    # Create environment matching large training settings
    env = create_financial_environment(
        data_list_filename=data_list_file,
        ticker_hash_file=ticker_hash_file,
        use_flat_obs=True,
        discrete_actions=200,
        start_date_idx=0,
        end_date_idx_plus1=10,  # More days for debugging
        action_update_interval=1,
        transaction_cost_ratio=0.0015,
        device='cuda'
    )
    
    print("=== Portfolio Value Debug Test ===")
    print(f"Environment: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\nInitial state:")
    print(f"  Initial X: {info.get('X', 'N/A')}")
    
    # Take a few actions and see what happens
    for step in range(3):
        print(f"\n--- Step {step + 1} ---")
        
        # Take a random action
        action = np.random.randint(0, env.action_space.n)
        print(f"Taking action: {action}")
        
        # Step environment
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        print(f"Reward: {reward:.6f}")
        print(f"Portfolio X: {step_info.get('X', 'N/A'):.6f}")
        print(f"Sharpe: {step_info.get('sharpe', 'N/A')}")
        
        obs = next_obs
        
        if terminated or truncated:
            print("Episode ended")
            break
    
    env.close()

if __name__ == "__main__":
    test_portfolio_debug()