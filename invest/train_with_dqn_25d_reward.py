#!/usr/bin/env python3
"""
DQN Training with MODIFIED 25-DAY REWARD FUNCTION

This module provides training functions for DQN agents using the modified 
financial environment that rewards 25-day series returns instead of 
immediate portfolio value changes.

Key change: reward = actual_return / stddev (25-day) vs (X_new - X_old)/X_old / stddev (immediate)
"""

import sys
import os
import time
import torch
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

# Import the modified environment
from financial_env_25d_reward import FinancialEnvironmentWrapper25D
from financial_dqn_agent import FinancialDQNAgent


def train_financial_dqn_25d_reward(
    data_list_filename: str,
    ticker_hash_file: str,
    exp_id: str,
    start_date_idx: int,
    end_date_idx_plus1: int,
    eval_start_date_idx: int,
    eval_end_date_idx_plus1: int,
    num_episodes: int = 10,
    num_discrete_actions: int = 200,
    gamma: float = 0.99,
    lr: float = 0.0001,
    batch_size: int = 64,
    memory_size: int = 20000,
    epsilon_start: float = 0.9,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.995,
    target_update_frequency: int = 100,
    action_update_interval: int = 10,
    transaction_cost_ratio: float = 0.0015,
    log_interval: int = 5,
    save_interval: int = 20,
    device: str = 'cpu',
    seed: int = 42
) -> Tuple[FinancialDQNAgent, Dict]:
    """
    Train DQN agent with 25-day reward function.
    
    Key difference: Uses FinancialEnvironmentWrapper25D which rewards
    25-day series returns instead of immediate portfolio changes.
    """
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n🎯 TRAINING WITH 25-DAY REWARD FUNCTION")
    print(f"Reward: actual_return / stddev (25-day series)")
    print(f"vs Original: (X_new - X_old)/X_old / stddev (immediate)")
    
    # Create modified environment
    env = FinancialEnvironmentWrapper25D(
        data_list_filename=data_list_filename,
        ticker_hash_file=ticker_hash_file,
        start_date_idx=start_date_idx,
        end_date_idx_plus1=end_date_idx_plus1,
        action_update_interval=action_update_interval,
        transaction_cost_ratio=transaction_cost_ratio,
        device=device,
        mode='dqn',
        num_discrete_actions=num_discrete_actions
    )
    
    print(f"Environment created:")
    print(f"  Observation dim: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Create DQN agent
    agent = FinancialDQNAgent(
        observation_dim=env.observation_space[0],
        n_actions=env.action_space,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        memory_size=memory_size,
        batch_size=batch_size,
        target_update_frequency=target_update_frequency,
        device=device
    )
    
    print(f"Financial DQN Agent initialized:")
    print(f"  Observation dim: {env.observation_space[0]}")
    print(f"  Actions: {env.action_space}")
    print(f"  Memory: Standard")
    print(f"  Device: {device}")
    print(f"  TD3 Features:")
    print(f"    Twin networks: False")
    print(f"    Tau updates: False (tau=N/A)")
    print(f"    Policy delay: 2")
    print(f"    Dueling: True")
    print("DQN Agent created with dueling architecture")
    
    # Training history
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'episode_portfolio_values': [],
        'episode_losses': [],
        'epsilon_history': [],
        'reward_components': [],  # Track both reward types
        'training_steps': 0
    }
    
    # Create save directory
    save_dir = f"/home/ubuntu/code/angle_rl/invest/data/{exp_id}/dqn"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n🚀 STARTING TRAINING LOOP 🚀")
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {end_date_idx_plus1 - start_date_idx}")
    print(f"Total training steps: {num_episodes * (end_date_idx_plus1 - start_date_idx)}")
    print(f"Progress logging every {log_interval} episodes")
    print("=" * 60)
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode:4d} STARTED ===")
        
        # Reset environment
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_losses = []
        episode_reward_components = []
        
        # Get initial portfolio value (X is in the observation)
        print(f"Episode {episode:4d} | Initial Portfolio Value: {obs[-252]:.6f}")  # X is near the end
        
        step_count = 0
        
        while True:
            # Select action
            action = agent.select_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            step_count += 1
            
            # Store reward components for analysis
            reward_info = {
                'step': step_count,
                '25d_reward': reward,  # This is the new 25-day reward
                'portfolio_reward': info.get('original_portfolio_reward', 0),  # Original reward for comparison
                'portfolio_value': info.get('portfolio_value', 1.0),
                'actual_return': info.get('actual_return', 0),
                'stddev': info.get('stddev', 1.0)
            }
            episode_reward_components.append(reward_info)
            
            # Store transition (using correct method name)
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train agent
            if agent.can_train():
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)
                    history['training_steps'] += 1
            
            # Log progress during episode
            if step_count % 50 == 0:
                current_portfolio = info.get('portfolio_value', 1.0)
                portfolio_return = (current_portfolio - 1.0) * 100
                print(f"    Step {step_count:4d} | Portfolio: {current_portfolio:.6f} ({portfolio_return:+.2f}%) | Reward: {reward:.4f} | Epsilon: {agent.epsilon:.3f}")
                
                # Show training loss if available
                if episode_losses and step_count % 100 == 0:
                    recent_loss = np.mean(episode_losses[-10:]) if len(episode_losses) >= 10 else np.mean(episode_losses)
                    print(f"    Step {step_count:4d} | Training | Losses: {{'q1_loss': {recent_loss}}}")
            
            obs = next_obs
            
            if done:
                break
        
        # Get final portfolio value
        final_portfolio = episode_reward_components[-1]['portfolio_value'] if episode_reward_components else 1.0
        final_return = (final_portfolio - 1.0) * 100
        
        print(f"Episode {episode:4d} | Final Portfolio Value: {final_portfolio:.6f} | Return: {final_return:.2f}%")
        
        print(f"=== Episode {episode:4d} FINISHED | Total Reward: {episode_reward:.4f} | Steps: {episode_length} ===")
        
        # Store episode results
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        history['episode_portfolio_values'].append(final_portfolio)
        history['episode_losses'].append(episode_losses)
        history['epsilon_history'].append(agent.epsilon)
        history['reward_components'].append(episode_reward_components)
        
        # Log progress summary
        if (episode + 1) % log_interval == 0:
            recent_episodes = min(log_interval, len(history['episode_rewards']))
            avg_reward = np.mean(history['episode_rewards'][-recent_episodes:])
            avg_length = np.mean(history['episode_lengths'][-recent_episodes:])
            avg_portfolio = np.mean(history['episode_portfolio_values'][-recent_episodes:])
            std_portfolio = np.std(history['episode_portfolio_values'][-recent_episodes:])
            
            print(f"\n{'='*80}")
            print(f"PROGRESS SUMMARY - Episodes {episode-recent_episodes+1} to {episode}")
            print(f"{'='*80}")
            print(f"Avg Reward: {avg_reward:8.3f} | Avg Length: {avg_length:6.1f} | Epsilon: {agent.epsilon:.3f}")
            print(f"Portfolio Stats: Avg={avg_portfolio:.6f}, Min={np.min(history['episode_portfolio_values'][-recent_episodes:]):.6f}, Max={np.max(history['episode_portfolio_values'][-recent_episodes:]):.6f}, Std={std_portfolio:.6f}")
            print(f"Memory Size: {len(agent.replay_buffer)} | Training Steps: {history['training_steps']}")
            
            # Skip Q-value logging for now (method compatibility issue)
            print(f"{'='*80}")
            print()
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(save_dir, f"dqn_episode_{episode+1}.pth")
            agent.save(checkpoint_path)
            
            # Save training history
            history_path = os.path.join(save_dir, f"training_history_episode_{episode+1}.pkl")
            with open(history_path, 'wb') as f:
                pickle.dump(history, f)
    
    print("Training completed!")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "dqn_final.pth")
    agent.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Save final training history
    final_history_path = os.path.join(save_dir, "training_history.pkl")
    with open(final_history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to {final_history_path}")
    
    return agent, history


def evaluate_financial_dqn_25d_reward(
    agent: FinancialDQNAgent,
    data_list_filename: str,
    ticker_hash_file: str,
    eval_start_date_idx: int,
    eval_end_date_idx_plus1: int,
    num_discrete_actions: int = 200,
    action_update_interval: int = 10,
    transaction_cost_ratio: float = 0.0015,
    device: str = 'cpu',
    online_learning: bool = False,
    eval_epsilon: float = 0.0
) -> Dict:
    """
    Evaluate DQN agent with 25-day reward function.
    
    Key difference: Uses FinancialEnvironmentWrapper25D for consistent reward calculation.
    """
    
    print(f"\n🎯 EVALUATION WITH 25-DAY REWARD FUNCTION")
    
    # Create evaluation environment
    eval_env = FinancialEnvironmentWrapper25D(
        data_list_filename=data_list_filename,
        ticker_hash_file=ticker_hash_file,
        start_date_idx=eval_start_date_idx,
        end_date_idx_plus1=eval_end_date_idx_plus1,
        action_update_interval=action_update_interval,
        transaction_cost_ratio=transaction_cost_ratio,
        device=device,
        mode='dqn',
        num_discrete_actions=num_discrete_actions
    )
    
    # Set evaluation mode
    original_epsilon = agent.epsilon
    if not online_learning:
        agent.epsilon = eval_epsilon
    
    if online_learning:
        print(f"Evaluation with ONLINE LEARNING: epsilon={eval_epsilon}")
        agent.epsilon = eval_epsilon
    else:
        print(f"Evaluation in DETERMINISTIC mode: epsilon=0")
        agent.epsilon = 0.0
    
    # Run evaluation episode
    obs = eval_env.reset()
    total_reward = 0
    episode_length = 0
    portfolio_values = []
    reward_components = []
    training_steps_during_eval = 0
    q_value_tracking = []
    
    # Initialize Q-value tracking (skip initial Q-values for now)
    q_value_tracking.append({
        'step': 0,
        'max_q': 0.0,
        'mean_q': 0.0,
        'min_q': 0.0
    })
    
    step = 0
    while True:
        # Select action using select_action method
        action = agent.select_action(obs)
        
        # Take step
        next_obs, reward, done, info = eval_env.step(action)
        
        total_reward += reward
        episode_length += 1
        step += 1
        
        # Store portfolio value
        portfolio_value = info.get('portfolio_value', 1.0)
        portfolio_values.append(portfolio_value)
        
        # Store reward components
        reward_info = {
            'step': step,
            '25d_reward': reward,
            'portfolio_reward': info.get('original_portfolio_reward', 0),
            'portfolio_value': portfolio_value,
            'actual_return': info.get('actual_return', 0),
            'stddev': info.get('stddev', 1.0)
        }
        reward_components.append(reward_info)
        
        # Online learning during evaluation
        if online_learning:
            agent.store_experience(obs, action, reward, next_obs, done)
            if agent.can_train():
                loss = agent.train_step()
                if loss is not None:
                    training_steps_during_eval += 1
        
        # Track Q-values periodically (simplified for now)
        if step % 10 == 0:
            q_value_tracking.append({
                'step': step,
                'max_q': 0.0,
                'mean_q': 0.0,
                'min_q': 0.0
            })
        
        obs = next_obs
        
        if done:
            break
    
    # Calculate final metrics
    final_portfolio_value = portfolio_values[-1] if portfolio_values else 1.0
    total_return = (final_portfolio_value - 1.0) * 100
    
    # Calculate Sharpe ratio from portfolio returns
    portfolio_returns = []
    for i in range(1, len(portfolio_values)):
        ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
        portfolio_returns.append(ret)
    
    if portfolio_returns:
        avg_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        avg_sharpe = avg_return / (std_return + 1e-10)
    else:
        avg_return = 0
        std_return = 0
        avg_sharpe = 0
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    results = {
        'total_reward': total_return,  # Convert to percentage for consistency
        'episode_length': episode_length,
        'final_portfolio_value': final_portfolio_value,
        'avg_return': avg_return,
        'std_return': std_return,
        'avg_sharpe': avg_sharpe,
        'portfolio_values': portfolio_values,
        'reward_components': reward_components,
        'q_value_tracking': q_value_tracking,
        'training_steps_during_eval': training_steps_during_eval,
        'reward_type': '25d_series_return'
    }
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Episode length: {episode_length}")
    print(f"Final portfolio value: {final_portfolio_value:.4f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Average Sharpe ratio: {avg_sharpe:.4f}")
    
    if online_learning:
        print(f"\n=== Online Learning Stats ===")
        print(f"Training steps during evaluation: {training_steps_during_eval}")
        print(f"Q-value tracking points: {len(q_value_tracking)}")
        if q_value_tracking:
            initial_q = q_value_tracking[0]
            final_q = q_value_tracking[-1]
            print(f"Initial Q-values: max={initial_q['max_q']:.4f}, mean={initial_q['mean_q']:.4f}")
            print(f"Final Q-values: max={final_q['max_q']:.4f}, mean={final_q['mean_q']:.4f}")
    
    return results