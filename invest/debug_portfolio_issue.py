#!/usr/bin/env python3
"""
Debug why portfolio values stay at 1.0 in large-scale training
"""

import sys
import torch
import numpy as np

sys.path.append('/home/ubuntu/code/angle_rl/invest')

from financial_env import create_financial_environment
from financial_dqn_agent import create_financial_dqn_agent

def test_portfolio_large_training():
    """Test with exact same settings as large training."""
    
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/dqn_large_exp_ticker_hash.pkl"
    
    # Create environment with EXACT same settings as large training
    env = create_financial_environment(
        data_list_filename=data_list_file,
        ticker_hash_file=ticker_hash_file,
        use_flat_obs=True,
        discrete_actions=200,
        start_date_idx=0,
        end_date_idx_plus1=265,  # Full 265 days like large training
        action_update_interval=1,
        transaction_cost_ratio=0.0015,
        device='cuda'
    )
    
    print("=== Large Training Portfolio Debug ===")
    print(f"Environment: {env.observation_space.shape[0]} obs, {env.action_space.n} actions")
    print(f"Training on {265} days")
    
    # Reset environment
    obs, info = env.reset()
    print(f"\n1. Initial state:")
    print(f"   Initial X: {info.get('X', 'N/A')}")
    
    portfolio_values = [info.get('X', 1.0)]
    
    # Take several steps and track portfolio changes
    for step in range(10):  # Just first 10 steps for debugging
        action = np.random.randint(0, env.action_space.n)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        
        current_X = step_info.get('X', 'N/A')
        portfolio_values.append(current_X)
        
        print(f"\n{step+2:2d}. Step {step+1}:")
        print(f"   Action: {action}")
        print(f"   Portfolio X: {current_X:.8f}")
        print(f"   Change: {(current_X - portfolio_values[-2]):.8f}")
        print(f"   Reward: {reward:.6f}")
        print(f"   Sharpe: {step_info.get('sharpe', 'N/A')}")
        
        obs = next_obs
        
        if terminated or truncated:
            print("Episode ended early")
            break
    
    print(f"\n=== Summary ===")
    print(f"Portfolio value range: {min(portfolio_values):.8f} to {max(portfolio_values):.8f}")
    print(f"Total change: {(portfolio_values[-1] - portfolio_values[0]):.8f}")
    print(f"Std deviation: {np.std(portfolio_values):.8f}")
    
    # Test a full episode to see final value
    print(f"\n=== Testing Full Episode ===")
    obs, info = env.reset()
    initial_X = info.get('X', 1.0)
    
    for step in range(265):  # Full episode
        action = np.random.randint(0, env.action_space.n)
        obs, reward, terminated, truncated, step_info = env.step(action)
        
        current_X = step_info.get('X', 1.0)
        
        if step % 50 == 0 or step >= 260:  # Log every 50 steps + last 5 steps
            print(f"   Step {step+1:3d}: X = {current_X:.8f}, Change = {(current_X - initial_X):.8f}, Term={terminated}, Trunc={truncated}")
        
        if terminated or truncated:
            print(f"   Episode ended at step {step+1}")
            break
    
    final_X = step_info.get('X', 1.0)
    print(f"\nFinal Results:")
    print(f"   Initial X: {initial_X:.8f}")
    print(f"   Final X:   {final_X:.8f}")
    print(f"   Total return: {((final_X / initial_X) - 1) * 100:.4f}%")
    
    env.close()

if __name__ == "__main__":
    test_portfolio_large_training()