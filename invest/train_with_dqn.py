"""
Training script that uses angle/RL's DQN infrastructure with financial environments.
This leverages the existing DQN implementation from the gaming repo.
"""

import sys
import os
import torch
import numpy as np
import pickle
from collections import deque
from typing import Dict, List, Tuple

# Add angle/RL to path
sys.path.append('/home/ubuntu/code/angle/RL')

# Import our financial DQN agent
from financial_dqn_agent import create_financial_dqn_agent

# Import financial environment
from financial_env import create_financial_environment


def train_financial_dqn(
    data_list_filename: str,
    ticker_hash_file: str,
    exp_id: str,
    start_date_idx: int = 0,
    end_date_idx_plus1: int = 267,
    eval_start_date_idx: int = 267,
    eval_end_date_idx_plus1: int = 367,
    # DQN parameters
    num_episodes: int = 100,
    num_discrete_actions: int = 100,
    gamma: float = 0.8,
    lr: float = 0.001,
    batch_size: int = 32,
    memory_size: int = 10000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    target_update_frequency: int = 10,
    # TD3 parameters
    use_twin_networks: bool = False,
    use_tau_updates: bool = False,
    tau: float = 0.005,
    policy_delay: int = 2,
    # Environment parameters
    action_update_interval: int = 10,
    transaction_cost_ratio: float = 0.0015,
    # Training parameters
    log_interval: int = 10,
    save_interval: int = 50,
    device: str = 'cuda',
    seed: int = 1,
):
    """
    Train DQN on financial data using angle/RL infrastructure.
    """
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Setup directories
    save_dir = f'/home/ubuntu/code/angle_rl/invest/data/{exp_id}/dqn/'
    os.makedirs(save_dir, exist_ok=True)
    
    # Create environment in DQN mode
    env = create_financial_environment(
        data_list_filename=data_list_filename,
        ticker_hash_file=ticker_hash_file,
        use_flat_obs=True,
        discrete_actions=num_discrete_actions,
        start_date_idx=start_date_idx,
        end_date_idx_plus1=end_date_idx_plus1,
        action_update_interval=action_update_interval,
        transaction_cost_ratio=transaction_cost_ratio,
        gamma=gamma,
        device=device
    )
    
    print(f"Environment created:")
    print(f"  Observation dim: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")
    
    # Create DQN agent
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n
    
    agent = create_financial_dqn_agent(
        observation_dim=obs_dim,
        n_actions=n_actions,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        lr=lr,
        batch_size=batch_size,
        memory_size=memory_size,
        target_update_frequency=target_update_frequency,
        use_dueling=True,
        use_prioritized_replay=False,
        # TD3 parameters
        use_twin_networks=use_twin_networks,
        use_tau_updates=use_tau_updates,
        tau=tau,
        policy_delay=policy_delay,
        device=device
    )
    
    td3_status = "with TD3 features" if (use_twin_networks or use_tau_updates) else "standard"
    print(f"DQN Agent created with dueling architecture ({td3_status})")
    if use_twin_networks or use_tau_updates:
        print(f"  TD3 Features: twin_networks={use_twin_networks}, tau_updates={use_tau_updates}, policy_delay={policy_delay}")
    
    print(f"\n🚀 STARTING TRAINING LOOP 🚀")
    print(f"Episodes: {num_episodes}")
    print(f"Steps per episode: {end_date_idx_plus1 - start_date_idx}")
    print(f"Total training steps: {num_episodes * (end_date_idx_plus1 - start_date_idx)}")
    print(f"Progress logging every {log_interval} episodes")
    print(f"="*60)
    
    # Training loop
    episode_rewards = []
    episode_lengths = []
    episode_portfolio_values = []
    
    for episode in range(num_episodes):
        # Episode start logging
        print(f"\n=== Episode {episode:4d} STARTED ===")
        
        # Reset environment
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        initial_portfolio = info.get('X', 1.0)
        
        print(f"Episode {episode:4d} | Initial Portfolio Value: {initial_portfolio:.6f}")
        
        # Get initial Q values for logging
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            obs_tensor = agent.devmgr.to_dev(obs_tensor)
            initial_q_values = agent.network(obs_tensor)
            max_q = initial_q_values.max().item()
            min_q = initial_q_values.min().item()
            mean_q = initial_q_values.mean().item()
        
        print(f"Episode {episode:4d} | Initial Q-values: Max={max_q:.4f}, Min={min_q:.4f}, Mean={mean_q:.4f}")
        
        # Episode loop
        done = False
        step_count = 0
        total_step_count = episode * (end_date_idx_plus1 - start_date_idx) + step_count
        
        while not done:
            # Select action
            action = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(action)
            done = terminated or truncated
            step_count += 1
            total_step_count += 1
            
            # Store transition
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train agent
            loss = None
            if len(agent.memory) >= agent.batch_size:
                loss = agent.train()
            
            # Step-level logging every 50 steps
            if total_step_count % 50 == 0:
                current_portfolio = step_info.get('X', 1.0)
                current_return = (current_portfolio - initial_portfolio) / initial_portfolio * 100
                print(f"    Step {total_step_count:4d} | Portfolio: {current_portfolio:.6f} ({current_return:+.2f}%) | Reward: {reward:.4f} | Epsilon: {agent.epsilon:.3f}")
                if loss:
                    loss_str = f" | Loss: {loss:.6f}" if isinstance(loss, (int, float)) else f" | Losses: {loss}"
                    print(f"    Step {total_step_count:4d} | Training{loss_str}")
            
            # Update state
            obs = next_obs
            episode_reward += reward
            episode_length += 1
        
        # Episode end logging
        final_portfolio = step_info.get('X', 1.0)
        portfolio_return = (final_portfolio - initial_portfolio) / initial_portfolio * 100
        
        # Get final Q values for logging
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            obs_tensor = agent.devmgr.to_dev(obs_tensor)
            final_q_values = agent.network(obs_tensor)
            final_max_q = final_q_values.max().item()
            final_min_q = final_q_values.min().item()
            final_mean_q = final_q_values.mean().item()
        
        print(f"Episode {episode:4d} | Final Portfolio Value: {final_portfolio:.6f} | Return: {portfolio_return:+.2f}%")
        print(f"Episode {episode:4d} | Final Q-values: Max={final_max_q:.4f}, Min={final_min_q:.4f}, Mean={final_mean_q:.4f}")
        print(f"=== Episode {episode:4d} FINISHED | Total Reward: {episode_reward:.4f} | Steps: {episode_length} ===")
        
        # Update epsilon
        agent.update_epsilon()
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if 'X' in step_info:
            episode_portfolio_values.append(step_info['X'])
        
        # Interval logging summary
        if episode % log_interval == 0:
            avg_reward = np.mean(episode_rewards[-log_interval:]) if len(episode_rewards) >= log_interval else episode_reward
            avg_length = np.mean(episode_lengths[-log_interval:]) if len(episode_lengths) >= log_interval else episode_length
            avg_portfolio = np.mean(episode_portfolio_values[-log_interval:]) if len(episode_portfolio_values) >= log_interval else 1.0
            
            # Calculate portfolio statistics
            recent_portfolios = episode_portfolio_values[-log_interval:] if len(episode_portfolio_values) >= log_interval else episode_portfolio_values
            if recent_portfolios:
                min_portfolio = min(recent_portfolios)
                max_portfolio = max(recent_portfolios)
                portfolio_std = np.std(recent_portfolios) if len(recent_portfolios) > 1 else 0.0
            else:
                min_portfolio = max_portfolio = portfolio_std = 1.0
            
            print(f"\n{'='*80}")
            print(f"PROGRESS SUMMARY - Episodes {max(0, episode-log_interval+1)} to {episode}")
            print(f"{'='*80}")
            print(f"Avg Reward: {avg_reward:8.3f} | Avg Length: {avg_length:6.1f} | Epsilon: {agent.epsilon:5.3f}")
            print(f"Portfolio Stats: Avg={avg_portfolio:.6f}, Min={min_portfolio:.6f}, Max={max_portfolio:.6f}, Std={portfolio_std:.6f}")
            print(f"Memory Size: {len(agent.memory):,} | Training Steps: {agent.training_steps:,}")
            
            # Current Q-value statistics
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                obs_tensor = agent.devmgr.to_dev(obs_tensor)
                current_q_values = agent.network(obs_tensor)
                current_max_q = current_q_values.max().item()
                current_min_q = current_q_values.min().item()
                current_mean_q = current_q_values.mean().item()
            
            print(f"Current Q-values: Max={current_max_q:.4f}, Min={current_min_q:.4f}, Mean={current_mean_q:.4f}")
            print(f"{'='*80}\n")
        
        # Save model
        if episode % save_interval == 0 and episode > 0:
            save_path = os.path.join(save_dir, f'dqn_episode_{episode}.pth')
            torch.save({
                'network': agent.network.state_dict(),
                'target_network': agent.target_network.state_dict(),
                'optimizer': agent.optimizer.state_dict(),
                'episode': episode,
                'epsilon': agent.epsilon,
                'total_steps': agent.total_steps
            }, save_path)
            print(f"Model saved to {save_path}")
    
    # Save final model and training stats
    final_save_path = os.path.join(save_dir, 'dqn_final.pth')
    torch.save({
        'network': agent.network.state_dict(),
        'target_network': agent.target_network.state_dict(),
        'optimizer': agent.optimizer.state_dict(),
        'episode': num_episodes,
        'epsilon': agent.epsilon,
        'total_steps': agent.total_steps
    }, final_save_path)
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_portfolio_values': episode_portfolio_values,
        'config': {
            'gamma': gamma,
            'lr': lr,
            'batch_size': batch_size,
            'memory_size': memory_size,
            'epsilon_start': epsilon_start,
            'epsilon_end': epsilon_end,
            'epsilon_decay': epsilon_decay,
            'target_update_frequency': target_update_frequency,
            'device': device
        },
        'env_params': {
            'start_date_idx': start_date_idx,
            'end_date_idx_plus1': end_date_idx_plus1,
            'action_update_interval': action_update_interval,
            'transaction_cost_ratio': transaction_cost_ratio,
            'num_discrete_actions': num_discrete_actions
        }
    }
    
    history_path = os.path.join(save_dir, 'training_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"Training completed!")
    print(f"Final model saved to {final_save_path}")
    print(f"Training history saved to {history_path}")
    
    # Cleanup
    env.close()
    
    return agent, history


def evaluate_financial_dqn(
    agent,  # FinancialDQNAgent
    data_list_filename: str,
    ticker_hash_file: str,
    eval_start_date_idx: int,
    eval_end_date_idx_plus1: int,
    num_discrete_actions: int = 100,
    action_update_interval: int = 10,
    transaction_cost_ratio: float = 0.0015,
    device: str = 'cuda',
    online_learning: bool = True,  # NEW: Enable learning during evaluation
    eval_epsilon: float = 0.1,     # NEW: Small exploration during evaluation
):
    """
    Evaluate trained DQN agent on financial data.
    
    Args:
        online_learning: If True, continue updating Q-networks during evaluation.
                        If False, use fixed policy (traditional evaluation).
        eval_epsilon: Exploration rate during evaluation (only if online_learning=True).
    """
    
    # Create evaluation environment
    env = create_financial_environment(
        data_list_filename=data_list_filename,
        ticker_hash_file=ticker_hash_file,
        use_flat_obs=True,
        discrete_actions=num_discrete_actions,
        start_date_idx=eval_start_date_idx,
        end_date_idx_plus1=eval_end_date_idx_plus1,
        action_update_interval=action_update_interval,
        transaction_cost_ratio=transaction_cost_ratio,
        device=device
    )
    
    # Configure agent for evaluation
    if online_learning:
        # Continue learning during evaluation (more realistic for financial RL)
        agent.network.train()
        agent.epsilon = eval_epsilon  # Small exploration to adapt to new data
        print(f"Evaluation with ONLINE LEARNING: epsilon={eval_epsilon}")
    else:
        # Traditional fixed policy evaluation
        agent.network.eval()
        agent.epsilon = 0.0  # No exploration during evaluation
        print(f"Evaluation with FIXED POLICY: epsilon=0.0")
    
    # Evaluation loop
    obs, info = env.reset()
    total_reward = 0.0
    episode_length = 0
    portfolio_values = [info.get('X', 1.0)]
    sharpe_ratios = []
    returns = []
    
    # Track learning during evaluation
    eval_losses = []
    q_value_stats = []
    
    done = False
    step_count = 0
    while not done:
        # Select action
        if online_learning:
            # With small exploration for adaptation
            action = agent.select_action(obs)
        else:
            # Pure exploitation
            with torch.no_grad():
                action = agent.select_action(obs)
        
        # Step environment
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        
        # Online learning during evaluation
        if online_learning:
            # Store experience and learn from it
            agent.store_experience(obs, action, reward, next_obs, done)
            
            # Train if we have enough experiences
            if len(agent.memory) >= agent.batch_size:
                losses = agent.train()
                if losses:
                    eval_losses.append(losses)
            
            # Log Q-value evolution during evaluation
            if step_count % 10 == 0:  # Every 10 steps
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                    obs_tensor = agent.devmgr.to_dev(obs_tensor)
                    q_vals = agent.network(obs_tensor)
                    q_value_stats.append({
                        'step': step_count,
                        'max_q': q_vals.max().item(),
                        'min_q': q_vals.min().item(),
                        'mean_q': q_vals.mean().item()
                    })
        
        # Record stats
        total_reward += reward
        episode_length += 1
        step_count += 1
        
        if 'X' in step_info:
            portfolio_values.append(step_info['X'])
        if 'sharpe' in step_info:
            sharpe_ratios.append(step_info['sharpe'])
        if 'actual_return' in step_info:
            returns.append(step_info['actual_return'])
        
        obs = next_obs
    
    # Calculate evaluation metrics
    final_portfolio_value = portfolio_values[-1] if portfolio_values else 1.0
    total_return = (final_portfolio_value - 1.0) * 100  # Percentage
    avg_sharpe = np.mean(sharpe_ratios) if sharpe_ratios else 0.0
    
    print(f"\n=== Evaluation Results ===")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Episode length: {episode_length}")
    print(f"Final portfolio value: {final_portfolio_value:.4f}")
    print(f"Total return: {total_return:.2f}%")
    print(f"Average Sharpe ratio: {avg_sharpe:.4f}")
    
    if online_learning:
        print(f"\n=== Online Learning Stats ===")
        print(f"Training steps during evaluation: {len(eval_losses)}")
        print(f"Q-value tracking points: {len(q_value_stats)}")
        if q_value_stats:
            print(f"Initial Q-values: max={q_value_stats[0]['max_q']:.4f}, mean={q_value_stats[0]['mean_q']:.4f}")
            print(f"Final Q-values: max={q_value_stats[-1]['max_q']:.4f}, mean={q_value_stats[-1]['mean_q']:.4f}")
    
    # Cleanup
    env.close()
    
    return {
        'total_reward': total_reward,
        'episode_length': episode_length,
        'final_portfolio_value': final_portfolio_value,
        'total_return': total_return,
        'avg_sharpe': avg_sharpe,
        'portfolio_values': portfolio_values,
        'sharpe_ratios': sharpe_ratios,
        'returns': returns,
        'online_learning': online_learning,
        'eval_losses': eval_losses if online_learning else [],
        'q_value_evolution': q_value_stats if online_learning else []
    }


if __name__ == "__main__":
    # Example usage
    data_list_filename = "/home/ubuntu/code/angle_rl/invest/data/data_list.txt"
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/data/ticker_hash.pkl"
    exp_id = "dqn_experiment"
    
    # Check if files exist
    if not os.path.exists(data_list_filename):
        print(f"Data list file not found: {data_list_filename}")
        print("Please provide valid data files.")
        exit(1)
    
    if not os.path.exists(ticker_hash_file):
        print(f"Ticker hash file not found: {ticker_hash_file}")
        print("Please provide valid ticker hash file.")
        exit(1)
    
    # Train DQN
    agent, history = train_financial_dqn(
        data_list_filename=data_list_filename,
        ticker_hash_file=ticker_hash_file,
        exp_id=exp_id,
        num_episodes=10,  # Small number for testing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Evaluate
    eval_results = evaluate_financial_dqn(
        agent=agent,
        data_list_filename=data_list_filename,
        ticker_hash_file=ticker_hash_file,
        eval_start_date_idx=267,
        eval_end_date_idx_plus1=367,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )