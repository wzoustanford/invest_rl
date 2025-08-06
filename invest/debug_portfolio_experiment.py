#!/usr/bin/env python3
"""
Debug portfolio value changes and trading frequency.
"""

import sys
import numpy as np
import pickle
from financial_env import create_financial_environment
from financial_dqn_agent import FinancialDQNAgent

def debug_portfolio_experiment():
    print("=" * 80)
    print("DEBUG: Portfolio Value and Trading Frequency")
    print("=" * 80)
    
    # Load ticker hash
    try:
        ticker_hash_data = pickle.load(open('/home/ubuntu/code/angle_rl/invest/dqn_exp_ticker_hash.pkl', 'rb'))
        ticker_hash = ticker_hash_data['hash_D']
        num_tickers = ticker_hash_data['num_tickers']
        print(f"Loaded ticker hash with {num_tickers} tickers")
    except:
        print("WARNING: Could not load ticker hash")
        num_tickers = 100
    
    # Test different action update intervals
    intervals_to_test = [1, 5, 10]
    
    for interval in intervals_to_test:
        print(f"\n" + "=" * 60)
        print(f"Testing Action Update Interval: {interval} days")
        print("=" * 60)
        
        # Create environment with different intervals
        env = create_financial_environment(
            data_list_filename='/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt',
            ticker_hash_file='/home/ubuntu/code/angle_rl/invest/dqn_exp_ticker_hash.pkl',
            start_date_idx=0,
            end_date_idx_plus1=5,  # Use more files for longer episodes
            action_update_interval=interval,
            transaction_cost_ratio=0.0015,
            use_flat_obs=True,
            discrete_actions=num_tickers
        )
        
        # Create simple agent
        agent = FinancialDQNAgent(
            observation_dim=env.env.get_observation_dim(),
            n_actions=num_tickers,
            epsilon_start=0.5,  # More exploration
            epsilon_end=0.1,
            batch_size=8,  # Smaller batch
            memory_size=500
        )
        
        # Run one episode with detailed logging
        print(f"\n--- Episode with {interval}-day trading interval ---")
        
        obs_result = env.reset()
        if isinstance(obs_result, tuple):
            observation, _ = obs_result
        else:
            observation = obs_result
        
        if isinstance(observation, list):
            observation = np.array(observation)
        
        print(f"Step 0: Initial portfolio value = {float(env.env.state['X']):.6f}")
        print(f"        Current data idx = {env.env.current_data_idx}")
        print(f"        Action update allowed? {env.env.current_data_idx % interval == 0}")
        
        episode_reward = 0
        max_steps = 25  # Run longer to see changes
        
        for step in range(1, max_steps + 1):
            # Select action
            action = agent.select_action(observation)
            
            # Take step
            step_result = env.step(action)
            if len(step_result) == 5:
                next_observation, reward, done, truncated, info = step_result
            else:
                next_observation, reward, done, info = step_result
            
            # Convert observation
            if isinstance(next_observation, list):
                next_observation = np.array(next_observation)
            
            # Store experience
            agent.store_experience(observation, action, reward, next_observation, done)
            
            # Detailed logging
            portfolio_value = float(env.env.state['X'])
            current_idx = env.env.current_data_idx
            action_day = current_idx % interval == 0
            
            print(f"Step {step:2d}: Portfolio = {portfolio_value:.6f}, "
                  f"Reward = {reward:.6f}, "
                  f"Data idx = {current_idx}, "
                  f"Action day? {action_day}")
            
            # Check if action was actually applied
            if hasattr(env.env, 'state') and 'action' in env.env.state:
                action_tensor = env.env.state['action']
                if hasattr(action_tensor, 'sum'):
                    action_sum = float(action_tensor.sum())
                    print(f"        Action sum = {action_sum:.6f}")
            
            episode_reward += reward
            observation = next_observation
            
            if done:
                print(f"        Episode ended at step {step}")
                break
        
        print(f"Final portfolio value: {float(env.env.state['X']):.6f}")
        print(f"Total episode reward: {episode_reward:.6f}")
        print(f"Portfolio changed? {abs(float(env.env.state['X']) - 1.0) > 1e-6}")
        
        # Train a bit if we have experiences
        if len(agent.memory) >= agent.batch_size:
            for _ in range(5):
                losses = agent.train()
                if losses:
                    print(f"Training losses: {losses}")
    
    print(f"\n" + "=" * 80)
    print("Portfolio debug completed!")
    print("=" * 80)

if __name__ == "__main__":
    debug_portfolio_experiment()