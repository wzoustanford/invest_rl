"""
Run DQN training on real financial data (first 20 trading days).
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


def create_ticker_hash_for_experiment(num_training_files=265):
    """Create ticker hash for the training data files."""
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/dqn_large_exp_ticker_hash.pkl"
    
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
        exp_id="dqn_large_exp"
    )
    
    # The function saves to exp_id + '_ticker_hash.pkl'
    return ticker_hash_file


def main():
    """Run DQN training on real financial data with large dataset."""
    
    # Setup
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    exp_id = "dqn_large_scale_exp"
    
    # Large scale parameters
    train_days = 265  # Train on 265 files
    eval_days = 60    # Evaluate on 60 files
    
    print(f"=== Large Scale DQN Training ===")
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
    num_episodes = 120  # Balanced for thorough training with reasonable runtime
    num_discrete_actions = 200  # Keep same action granularity
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Train DQN
    print(f"\nStarting large-scale DQN training:")
    print(f"  Training days: 0-{train_days}")
    print(f"  Evaluation days: {train_days}-{train_days + eval_days}")
    print(f"  Episodes: {num_episodes}")
    print(f"  Discrete actions: {num_discrete_actions}")
    
    agent, history = train_financial_dqn(
        data_list_filename=data_list_file,
        ticker_hash_file=ticker_hash_file,
        exp_id=exp_id,
        start_date_idx=0,
        end_date_idx_plus1=train_days,
        eval_start_date_idx=train_days,
        eval_end_date_idx_plus1=train_days + eval_days,
        # DQN parameters
        num_episodes=num_episodes,
        num_discrete_actions=num_discrete_actions,
        gamma=0.99,  # Higher gamma for longer-term rewards
        lr=0.0001,
        batch_size=64,  # Larger batch size for better stability
        memory_size=50000,  # Much larger memory for long training
        epsilon_start=1.0,
        epsilon_end=0.01,  # Lower final epsilon
        epsilon_decay=0.9995,  # Slower decay for longer training
        target_update_frequency=500,  # Less frequent updates for stability
        # Environment parameters
        action_update_interval=10,  # Update portfolio every 10 days
        transaction_cost_ratio=0.0015,
        # Training parameters - Updated for progress monitoring
        log_interval=5,  # Log every 5 episodes for better monitoring
        save_interval=50,
        device=device,
        seed=42
    )
    
    # Print training summary
    print("\n=== Training Summary ===")
    if len(history['episode_rewards']) > 0:
        print(f"Final episode reward: {history['episode_rewards'][-1]:.4f}")
        print(f"Average reward (last 10 episodes): {np.mean(history['episode_rewards'][-10:]):.4f}")
        
        if len(history['episode_portfolio_values']) > 0:
            final_train_portfolio = history['episode_portfolio_values'][-1]
            print(f"Final training portfolio value: {final_train_portfolio:.4f}")
            print(f"Training return: {(final_train_portfolio - 1.0) * 100:.2f}%")
    
    # Evaluate on test period with online learning
    print("\n=== Evaluating on test period ===")
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
        online_learning=True,  # Enable online learning during evaluation
        eval_epsilon=0.05      # Small exploration during evaluation
    )
    
    print("\n=== Final Results ===")
    print(f"Training:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Final training return: {(history['episode_portfolio_values'][-1] - 1.0) * 100:.2f}%" if history['episode_portfolio_values'] else "N/A")
    
    print(f"\nEvaluation:")
    print(f"  Test days: {eval_days}")
    print(f"  Final portfolio value: {eval_results['final_portfolio_value']:.4f}")
    print(f"  Total return: {eval_results['total_return']:.2f}%")
    print(f"  Average Sharpe ratio: {eval_results['avg_sharpe']:.4f}")
    
    # Plot results if possible
    try:
        import matplotlib.pyplot as plt
        
        # Plot training rewards
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['episode_rewards'])
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        
        plt.subplot(1, 3, 2)
        if history['episode_portfolio_values']:
            plt.plot(history['episode_portfolio_values'])
            plt.title('Portfolio Value')
            plt.xlabel('Episode')
            plt.ylabel('Portfolio Value')
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        
        plt.subplot(1, 3, 3)
        if eval_results['portfolio_values']:
            plt.plot(eval_results['portfolio_values'])
            plt.title('Evaluation Portfolio Value')
            plt.xlabel('Day')
            plt.ylabel('Portfolio Value')
            plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f'/home/ubuntu/code/angle_rl/invest/data/{exp_id}/dqn/results.png')
        print(f"\nPlots saved to /home/ubuntu/code/angle_rl/invest/data/{exp_id}/dqn/results.png")
        
    except ImportError:
        print("\nMatplotlib not available, skipping plots")
    
    return agent, history, eval_results


if __name__ == "__main__":
    agent, history, eval_results = main()