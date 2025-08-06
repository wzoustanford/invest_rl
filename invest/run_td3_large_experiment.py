#!/usr/bin/env python3
"""
Run TD3 DQN training on 265 days training + 60 days evaluation with online learning.
"""

import sys
import os
import torch
import numpy as np
import pickle

# Add paths
sys.path.append('/home/ubuntu/code/angle_rl/invest')

# Import utilities and training functions
from utils import aggregate_tickers_RL
from train_with_dqn import train_financial_dqn, evaluate_financial_dqn
from financial_dqn_agent import create_financial_dqn_agent


def create_ticker_hash_for_experiment(num_training_files=265):
    """Create ticker hash for the training data files."""
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/td3_large_exp_ticker_hash.pkl"
    
    # Read the data file list first
    with open(data_list_file, 'r') as f:
        data_files = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"Found {len(data_files)} data files")
    print(f"Using first {num_training_files} files for ticker hash creation")
    
    # Create ticker hash for training period
    print(f"Creating ticker hash for first {num_training_files} trading days...")
    aggregate_tickers_RL(
        data_file_list=data_files,
        start_idx=0,
        end_idx_plus1=num_training_files,
        exp_id="td3_large_exp"
    )
    
    # The function saves to exp_id + '_ticker_hash.pkl'
    return ticker_hash_file


def main():
    """Run TD3 DQN training on real financial data with large dataset."""
    
    # Setup
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    exp_id = "td3_large_scale_exp"
    
    # Large scale parameters
    train_days = 265  # Train on 265 files
    eval_days = 60    # Evaluate on 60 files
    
    print(f"=== Large Scale TD3 DQN Training ===")
    print(f"Training files: {train_days}")
    print(f"Evaluation files: {eval_days}")
    print(f"Total files used: {train_days + eval_days}")
    
    # Create ticker hash
    ticker_hash_file = create_ticker_hash_for_experiment(num_training_files=train_days)
    
    # Load ticker hash to see how many stocks we have
    with open(ticker_hash_file, 'rb') as f:
        ticker_data = pickle.load(f)
    
    print(f"\nTicker hash created:")
    print(f"  Number of unique tickers: {ticker_data['num_tickers']}")
    print(f"  Hash keys: {list(ticker_data.keys())}")
    
    # Training parameters
    num_episodes = 50   # Fewer episodes for initial testing
    num_discrete_actions = 200
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create TD3 agent
    print(f"\n=== Creating TD3 Agent ===")
    from financial_env import create_financial_environment
    
    # Create temporary environment to get observation dimension
    temp_env = create_financial_environment(
        data_list_filename=data_list_file,
        ticker_hash_file=ticker_hash_file,
        use_flat_obs=True,
        discrete_actions=num_discrete_actions,
        start_date_idx=0,
        end_date_idx_plus1=5,  # Just for initialization
        action_update_interval=10,
        transaction_cost_ratio=0.0015,
        device=device
    )
    
    # Create TD3 agent with twin networks, tau updates, and policy delay
    agent = create_financial_dqn_agent(
        observation_dim=temp_env.env.get_observation_dim(),
        n_actions=ticker_data['num_tickers'],
        gamma=0.99,
        epsilon_start=0.3,  # Lower starting epsilon for TD3
        epsilon_end=0.01,
        epsilon_decay=0.9995,
        lr=0.0001,
        batch_size=64,
        memory_size=50000,
        target_update_frequency=500,
        use_dueling=True,
        # TD3 features
        use_twin_networks=True,   # Twin Q-networks
        use_tau_updates=True,     # Soft target updates
        tau=0.005,               # Tau parameter for soft updates
        policy_delay=2,          # Policy update delay
        device=device
    )
    
    temp_env.close()
    
    # Run manual training loop with TD3 agent
    print(f"\n=== Starting TD3 Training ===")
    print(f"TD3 Features Enabled:")
    print(f"  Twin Q-networks: {agent.use_twin_networks}")
    print(f"  Tau updates: {agent.use_tau_updates} (tau={agent.tau})")
    print(f"  Policy delay: {agent.policy_delay}")
    
    # Create training environment
    train_env = create_financial_environment(
        data_list_filename=data_list_file,
        ticker_hash_file=ticker_hash_file,
        use_flat_obs=True,
        discrete_actions=num_discrete_actions,
        start_date_idx=0,
        end_date_idx_plus1=train_days,
        action_update_interval=10,
        transaction_cost_ratio=0.0015,
        device=device
    )
    
    # Training loop
    episode_rewards = []
    episode_portfolio_values = []
    
    for episode in range(num_episodes):
        # Reset environment
        obs_result = train_env.reset()
        if isinstance(obs_result, tuple):
            obs, info = obs_result
        else:
            obs = obs_result
            info = {}
        
        if isinstance(obs, list):
            obs = np.array(obs)
        
        episode_reward = 0
        step_count = 0
        initial_portfolio = float(train_env.env.state['X'])
        
        done = False
        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Step environment
            step_result = train_env.step(action)
            if len(step_result) == 5:
                next_obs, reward, terminated, truncated, step_info = step_result
            else:
                next_obs, reward, terminated, step_info = step_result
                truncated = False
            
            done = terminated or truncated
            
            if isinstance(next_obs, list):
                next_obs = np.array(next_obs)
            
            # Store experience
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train TD3 agent
            if len(agent.memory) >= agent.batch_size:
                losses = agent.train()
                if losses and step_count % 50 == 0:
                    loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                    print(f"  Episode {episode}, Step {step_count}: {loss_str}")
            
            episode_reward += reward
            step_count += 1
            obs = next_obs
            
            # Update epsilon
            agent.update_epsilon()
        
        final_portfolio = float(train_env.env.state['X'])
        episode_rewards.append(episode_reward)
        episode_portfolio_values.append(final_portfolio)
        
        print(f"Episode {episode:3d}: Reward={episode_reward:8.4f}, "
              f"Portfolio: {initial_portfolio:.4f}→{final_portfolio:.4f} "
              f"({((final_portfolio/initial_portfolio-1)*100):+.2f}%), "
              f"Steps={step_count}, Epsilon={agent.epsilon:.3f}")
    
    train_env.close()
    
    # Evaluate on test period with online learning
    print(f"\n=== Evaluating with Online Learning ===")
    eval_results = evaluate_financial_dqn(
        agent=agent,
        data_list_filename=data_list_file,
        ticker_hash_file=ticker_hash_file,
        eval_start_date_idx=train_days,
        eval_end_date_idx_plus1=train_days + eval_days,
        num_discrete_actions=num_discrete_actions,
        action_update_interval=10,
        transaction_cost_ratio=0.0015,
        device=device,
        online_learning=True,  # Continue learning during evaluation
        eval_epsilon=0.05      # Small exploration during evaluation
    )
    
    print(f"\n=== TD3 Final Results ===")
    print(f"Training:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Final training portfolio: {episode_portfolio_values[-1]:.4f}")
    print(f"  Training return: {(episode_portfolio_values[-1] - 1.0) * 100:.2f}%")
    
    print(f"\nEvaluation (with online learning):")
    print(f"  Test days: {eval_days}")
    print(f"  Final portfolio value: {eval_results['final_portfolio_value']:.4f}")
    print(f"  Total return: {eval_results['total_return']:.2f}%")
    print(f"  Average Sharpe ratio: {eval_results['avg_sharpe']:.4f}")
    
    if eval_results['online_learning']:
        print(f"  Online learning steps: {len(eval_results['eval_losses'])}")
        print(f"  Q-value evolution tracked: {len(eval_results['q_value_evolution'])} points")
    
    return agent, episode_rewards, episode_portfolio_values, eval_results


if __name__ == "__main__":
    agent, history_rewards, history_portfolios, eval_results = main()