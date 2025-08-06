#!/usr/bin/env python3
"""
Test TD3 features in financial DQN agent with 2-3 episodes.
Observe portfolio values (X) and Q-values.
"""

import sys
import numpy as np
import pickle
from financial_env import create_financial_environment
from financial_dqn_agent import FinancialDQNAgent

def run_td3_experiment():
    print("=" * 80)
    print("TD3 Financial DQN Experiment")
    print("=" * 80)
    
    # Load data list
    with open('/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt', 'r') as f:
        data_files = [line.strip() for line in f.readlines()]
    
    # Use first few files for quick test
    test_files = data_files[:3]
    print(f"Using {len(test_files)} data files for testing")
    
    # Load ticker hash
    try:
        ticker_hash_data = pickle.load(open('/home/ubuntu/code/angle_rl/invest/dqn_exp_ticker_hash.pkl', 'rb'))
        ticker_hash = ticker_hash_data['hash_D']
        num_tickers = ticker_hash_data['num_tickers']
        print(f"Loaded ticker hash with {num_tickers} tickers")
    except:
        print("WARNING: Could not load ticker hash, using limited tickers")
        ticker_hash = None
        num_tickers = 100  # fallback
    
    # Create environment with wrapper for DQN compatibility
    env = create_financial_environment(
        data_list_filename='/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt',
        ticker_hash_file='/home/ubuntu/code/angle_rl/invest/dqn_exp_ticker_hash.pkl',
        start_date_idx=0,
        end_date_idx_plus1=3,  # Use first 3 files for testing
        action_update_interval=10,  # Trade every 10 days
        transaction_cost_ratio=0.0015,
        use_flat_obs=True,  # For DQN compatibility
        discrete_actions=num_tickers
    )
    
    # Test configurations
    configs = [
        {
            'name': 'Standard DQN',
            'use_twin_networks': False,
            'use_tau_updates': False,
            'policy_delay': 1
        },
        {
            'name': 'TD3 (Twin + Tau + Delay)',
            'use_twin_networks': True,
            'use_tau_updates': True,
            'tau': 0.005,
            'policy_delay': 2
        }
    ]
    
    for config in configs:
        print(f"\n" + "=" * 60)
        print(f"Testing: {config['name']}")
        print("=" * 60)
        
        # Create agent with TD3 features
        agent = FinancialDQNAgent(
            observation_dim=env.env.get_observation_dim(),
            n_actions=num_tickers,
            use_dueling=True,
            epsilon_start=0.3,  # Lower epsilon for testing
            epsilon_end=0.1,
            lr=0.001,
            batch_size=16,  # Smaller batch for quick testing
            memory_size=1000,
            target_update_frequency=50,
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        # Run episodes
        for episode in range(3):
            print(f"\n--- Episode {episode + 1} ---")
            
            obs_result = env.reset()  # Wrapper handles flat observations
            if isinstance(obs_result, tuple):
                observation, _ = obs_result  # Extract observation from tuple
            else:
                observation = obs_result
            episode_reward = 0
            step = 0
            
            # Debug observation shape
            print(f"  Observation type: {type(observation)}")
            print(f"  Observation shape: {np.array(observation).shape if isinstance(observation, (list, np.ndarray)) else 'N/A'}")
            
            # Get initial Q-values
            if isinstance(observation, list):
                observation = np.array(observation)
            q_values = agent.get_q_values(observation)
            print(f"Initial state:")
            print(f"  Portfolio Value (X): {float(env.env.state['X']):.6f}")
            if 'q1' in q_values:
                print(f"  Q1 values: max={q_values['q1'].max():.4f}, min={q_values['q1'].min():.4f}, mean={q_values['q1'].mean():.4f}")
            if 'q2' in q_values:
                print(f"  Q2 values: max={q_values['q2'].max():.4f}, min={q_values['q2'].min():.4f}, mean={q_values['q2'].mean():.4f}")
            if 'q_min' in q_values:
                print(f"  Q_min values: max={q_values['q_min'].max():.4f}, min={q_values['q_min'].min():.4f}, mean={q_values['q_min'].mean():.4f}")
            
            done = False
            max_steps = 50  # Limit steps for quick testing
            
            while not done and step < max_steps:
                # Select action
                action = agent.select_action(observation)
                
                # Take step
                step_result = env.step(action)
                if len(step_result) == 5:
                    next_observation, reward, done, truncated, info = step_result
                else:
                    next_observation, reward, done, info = step_result
                
                # Ensure observations are numpy arrays
                if isinstance(next_observation, list):
                    next_observation = np.array(next_observation)
                
                # Store experience
                agent.store_experience(observation, action, reward, next_observation, done)
                
                # Train if we have enough experiences
                if len(agent.memory) >= agent.batch_size:
                    losses = agent.train()
                    if losses and step % 10 == 0:
                        loss_str = ", ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
                        print(f"  Step {step}: {loss_str}")
                
                observation = next_observation
                episode_reward += reward
                step += 1
                
                # Update epsilon
                agent.update_epsilon()
            
            # Final state
            q_values = agent.get_q_values(observation)
            print(f"Final state:")
            print(f"  Portfolio Value (X): {float(env.env.state['X']):.6f}")
            print(f"  Episode Reward: {episode_reward:.6f}")
            print(f"  Total Steps: {step}")
            if 'q1' in q_values:
                print(f"  Q1 values: max={q_values['q1'].max():.4f}, min={q_values['q1'].min():.4f}, mean={q_values['q1'].mean():.4f}")
            if 'q2' in q_values:
                print(f"  Q2 values: max={q_values['q2'].max():.4f}, min={q_values['q2'].min():.4f}, mean={q_values['q2'].mean():.4f}")
            if 'q_min' in q_values:
                print(f"  Q_min values: max={q_values['q_min'].max():.4f}, min={q_values['q_min'].min():.4f}, mean={q_values['q_min'].mean():.4f}")
            
            print(f"  Epsilon: {agent.epsilon:.4f}")
            print(f"  Training steps: {agent.training_steps}")
    
    print(f"\n" + "=" * 80)
    print("TD3 Experiment completed!")
    print("=" * 80)

if __name__ == "__main__":
    run_td3_experiment()