#!/usr/bin/env python3
"""
Sliding Window Experiment for Financial DQN/TD3
Test multiple time windows to get ~240 evaluation days (~1 year of results)

Design:
- Window 1: Train[0:265], Eval[265:325] (60 eval days)
- Window 2: Train[60:325], Eval[325:385] (60 eval days)
- Window 3: Train[120:385], Eval[385:445] (60 eval days)
- Window 4: Train[180:445], Eval[445:505] (60 eval days)
Total: 240 evaluation days across 4 windows
"""

import sys
import os
import time
import json
import pickle
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple

# Add paths
sys.path.append('/home/ubuntu/code/angle_rl/invest')

# Import utilities and training functions
from utils import aggregate_tickers_RL
from train_with_dqn import train_financial_dqn, evaluate_financial_dqn
from financial_dqn_agent import create_financial_dqn_agent


class SlidingWindowExperiment:
    """Manages sliding window experiments for financial RL."""
    
    def __init__(self, 
                 data_list_file: str = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt",
                 base_exp_id: str = "sliding_window",
                 training_window_size: int = 265,
                 eval_window_size: int = 60,
                 window_shift: int = 60,
                 target_eval_days: int = 240,
                 device: str = None):
        
        self.data_list_file = data_list_file
        self.base_exp_id = base_exp_id
        self.training_window_size = training_window_size
        self.eval_window_size = eval_window_size
        self.window_shift = window_shift
        self.target_eval_days = target_eval_days
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate number of windows needed
        self.num_windows = target_eval_days // eval_window_size
        
        # Check data availability
        with open(data_list_file, 'r') as f:
            self.total_files = len([line.strip() for line in f.readlines() if line.strip()])
        
        # Calculate window specifications
        self.windows = self._calculate_windows()
        
        # Results storage
        self.results = {
            'experiment_config': {
                'training_window_size': training_window_size,
                'eval_window_size': eval_window_size,
                'window_shift': window_shift,
                'num_windows': self.num_windows,
                'target_eval_days': target_eval_days,
                'device': self.device,
                'total_files_available': self.total_files
            },
            'windows': [],
            'summary': {}
        }
    
    def _calculate_windows(self) -> List[Dict]:
        """Calculate window specifications."""
        windows = []
        
        for i in range(self.num_windows):
            # Each window shifts by window_shift
            train_start = i * self.window_shift
            train_end = train_start + self.training_window_size
            eval_start = train_end
            eval_end = eval_start + self.eval_window_size
            
            # Check if we have enough data
            if eval_end > self.total_files:
                print(f"Warning: Window {i+1} would need {eval_end} files but only {self.total_files} available")
                break
            
            windows.append({
                'window_id': i + 1,
                'train_start': train_start,
                'train_end': train_end,
                'eval_start': eval_start,
                'eval_end': eval_end,
                'train_days': train_end - train_start,
                'eval_days': eval_end - eval_start
            })
        
        return windows
    
    def create_ticker_hash(self, window_spec: Dict) -> str:
        """Create ticker hash for a specific window."""
        exp_id = f"{self.base_exp_id}_w{window_spec['window_id']}"
        ticker_hash_file = f"/home/ubuntu/code/angle_rl/invest/{exp_id}_ticker_hash.pkl"
        
        # Read data files
        with open(self.data_list_file, 'r') as f:
            data_files = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"  Creating ticker hash for window {window_spec['window_id']} (files {window_spec['train_start']}-{window_spec['train_end']})...")
        
        # Create ticker hash for this window's training period
        aggregate_tickers_RL(
            data_file_list=data_files,
            start_idx=window_spec['train_start'],
            end_idx_plus1=window_spec['train_end'],
            exp_id=exp_id
        )
        
        return ticker_hash_file
    
    def run_window_experiment(self, window_spec: Dict, algorithm: str = 'dqn') -> Dict:
        """Run experiment for a single window."""
        print(f"\n=== Window {window_spec['window_id']} Experiment ===")
        print(f"Training: files {window_spec['train_start']}-{window_spec['train_end']} ({window_spec['train_days']} days)")
        print(f"Evaluation: files {window_spec['eval_start']}-{window_spec['eval_end']} ({window_spec['eval_days']} days)")
        
        start_time = time.time()
        
        # Create ticker hash for this window
        ticker_hash_file = self.create_ticker_hash(window_spec)
        
        # Load ticker data
        with open(ticker_hash_file, 'rb') as f:
            ticker_data = pickle.load(f)
        
        print(f"  Ticker hash: {ticker_data['num_tickers']} unique tickers")
        
        # Create agent
        exp_id = f"{self.base_exp_id}_w{window_spec['window_id']}"
        
        print(f"  Algorithm: {algorithm.upper()}")
        
        # Training parameters (matching train_financial_dqn signature)
        training_params = {
            'data_list_filename': self.data_list_file,
            'ticker_hash_file': ticker_hash_file,
            'exp_id': exp_id,
            'start_date_idx': window_spec['train_start'],
            'end_date_idx_plus1': window_spec['train_end'],
            'eval_start_date_idx': window_spec['eval_start'],
            'eval_end_date_idx_plus1': window_spec['eval_end'],
            # DQN parameters
            'num_episodes': 5,  # Back to 5 episodes for TD3 comparison
            'num_discrete_actions': 200,
            'gamma': 0.99,
            'lr': 0.0001,
            'batch_size': 64,
            'memory_size': 20000,  # Smaller memory for faster training
            'epsilon_start': 0.3,  # Lower starting epsilon for faster convergence
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,  # Faster decay
            'target_update_frequency': 300,  # More frequent updates
            # Environment parameters
            'action_update_interval': 10,
            'transaction_cost_ratio': 0.0015,
            # Training parameters  
            'log_interval': 1,  # Log every episode
            'save_interval': 5,  # Save every 5 episodes
            'device': self.device,
            'seed': 42 + window_spec['window_id']  # Different seed per window
        }
        
        print(f"  Training parameters: episodes={training_params['num_episodes']}, gamma={training_params['gamma']}, lr={training_params['lr']}")
        
        try:
            # Train the agent
            print(f"  Starting training ({training_params['num_episodes']} episodes)...")
            train_start_time = time.time()
            
            agent, history = train_financial_dqn(**training_params)
            
            train_time = time.time() - train_start_time
            print(f"  Training completed in {train_time:.1f}s")
            
            # Evaluate
            print(f"  Starting evaluation ({window_spec['eval_days']} days)...")
            eval_start_time = time.time()
            
            eval_results = evaluate_financial_dqn(
                agent=agent,
                data_list_filename=self.data_list_file,
                ticker_hash_file=ticker_hash_file,
                eval_start_date_idx=window_spec['eval_start'],
                eval_end_date_idx_plus1=window_spec['eval_end'],
                num_discrete_actions=training_params['num_discrete_actions'],
                action_update_interval=training_params['action_update_interval'],
                transaction_cost_ratio=training_params['transaction_cost_ratio'],
                device=self.device,
                online_learning=True,
                eval_epsilon=0.05
            )
            
            eval_time = time.time() - eval_start_time
            print(f"  Evaluation completed in {eval_time:.1f}s")
            
            # Calculate returns
            if history['episode_portfolio_values']:
                final_train_portfolio = history['episode_portfolio_values'][-1]
                train_return = (final_train_portfolio - 1.0) * 100
            else:
                train_return = 0.0
            
            eval_return = eval_results['total_return']
            
            total_time = time.time() - start_time
            
            # Compile results
            window_results = {
                'window_spec': window_spec,
                'algorithm': algorithm,
                'training': {
                    'episodes': training_params['num_episodes'],
                    'final_portfolio': final_train_portfolio if history['episode_portfolio_values'] else 1.0,
                    'return_pct': train_return,
                    'time_seconds': train_time
                },
                'evaluation': {
                    'days': window_spec['eval_days'],
                    'final_portfolio': eval_results['final_portfolio_value'],
                    'return_pct': eval_return,
                    'sharpe_ratio': eval_results['avg_sharpe'],
                    'time_seconds': eval_time
                },
                'total_time_seconds': total_time,
                'ticker_count': ticker_data['num_tickers']
            }
            
            print(f"  Results: Train={train_return:.2f}%, Eval={eval_return:.2f}%, Sharpe={eval_results['avg_sharpe']:.4f}")
            print(f"  Total time: {total_time:.1f}s")
            
            return window_results
            
        except Exception as e:
            print(f"  ERROR in window {window_spec['window_id']}: {str(e)}")
            return {
                'window_spec': window_spec,
                'error': str(e),
                'total_time_seconds': time.time() - start_time
            }
    
    def run_full_experiment(self, algorithm: str = 'dqn') -> Dict:
        """Run the complete sliding window experiment."""
        print(f"=== Sliding Window Experiment ===")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Windows: {len(self.windows)}")
        print(f"Total evaluation days: {len(self.windows) * self.eval_window_size}")
        print(f"Device: {self.device}")
        
        experiment_start_time = time.time()
        
        # Run each window
        window_results = []
        total_eval_days = 0
        
        for window_spec in self.windows:
            result = self.run_window_experiment(window_spec, algorithm)
            window_results.append(result)
            
            if 'error' not in result:
                total_eval_days += window_spec['eval_days']
        
        experiment_time = time.time() - experiment_start_time
        
        # Calculate summary statistics
        successful_windows = [r for r in window_results if 'error' not in r]
        
        if successful_windows:
            eval_returns = [r['evaluation']['return_pct'] for r in successful_windows]
            train_returns = [r['training']['return_pct'] for r in successful_windows]
            
            summary = {
                'total_windows': len(self.windows),
                'successful_windows': len(successful_windows),
                'total_eval_days': total_eval_days,
                'experiment_time_seconds': experiment_time,
                'experiment_time_hours': experiment_time / 3600,
                'avg_time_per_window': experiment_time / len(self.windows),
                'evaluation_performance': {
                    'mean_return_pct': np.mean(eval_returns),
                    'std_return_pct': np.std(eval_returns),
                    'min_return_pct': np.min(eval_returns),
                    'max_return_pct': np.max(eval_returns),
                    'returns_list': eval_returns
                },
                'training_performance': {
                    'mean_return_pct': np.mean(train_returns),
                    'std_return_pct': np.std(train_returns),
                    'returns_list': train_returns
                }
            }
        else:
            summary = {
                'total_windows': len(self.windows),
                'successful_windows': 0,
                'error': 'No successful windows'
            }
        
        # Store results
        self.results['windows'] = window_results
        self.results['summary'] = summary
        self.results['experiment_metadata'] = {
            'algorithm': algorithm,
            'start_time': datetime.now().isoformat(),
            'total_time_seconds': experiment_time
        }
        
        # Save results
        results_file = f'/home/ubuntu/code/angle_rl/invest/{self.base_exp_id}_{algorithm}_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n=== Experiment Complete ===")
        print(f"Total time: {experiment_time/3600:.2f} hours")
        print(f"Successful windows: {len(successful_windows)}/{len(self.windows)}")
        print(f"Total evaluation days: {total_eval_days}")
        
        if successful_windows:
            print(f"Average evaluation return: {summary['evaluation_performance']['mean_return_pct']:.2f}% ± {summary['evaluation_performance']['std_return_pct']:.2f}%")
            print(f"Range: {summary['evaluation_performance']['min_return_pct']:.2f}% to {summary['evaluation_performance']['max_return_pct']:.2f}%")
        
        print(f"Results saved to: {results_file}")
        
        return self.results
    
    def print_experiment_plan(self):
        """Print the experiment plan for review."""
        print("=== Sliding Window Experiment Plan ===")
        print(f"Data file: {self.data_list_file}")
        print(f"Total files available: {self.total_files}")
        print(f"Training window size: {self.training_window_size} days")
        print(f"Evaluation window size: {self.eval_window_size} days")
        print(f"Window shift: {self.window_shift} days")
        print(f"Target evaluation days: {self.target_eval_days}")
        print(f"Calculated windows: {len(self.windows)}")
        print(f"Device: {self.device}")
        
        print(f"\nWindow Details:")
        for window in self.windows:
            print(f"  Window {window['window_id']}: "
                  f"Train[{window['train_start']}:{window['train_end']}] → "
                  f"Eval[{window['eval_start']}:{window['eval_end']}]")
        
        total_eval_days = len(self.windows) * self.eval_window_size
        print(f"\nTotal evaluation days: {total_eval_days}")
        print(f"Estimated time per window: ~15-30 minutes")
        print(f"Estimated total time: {len(self.windows) * 20:.0f}-{len(self.windows) * 30:.0f} minutes")


def main():
    """Main function to run sliding window experiment."""
    
    # Create experiment
    experiment = SlidingWindowExperiment(
        target_eval_days=240,  # ~1 year of evaluation
        training_window_size=265,
        eval_window_size=60,
        window_shift=60
    )
    
    # Print plan
    experiment.print_experiment_plan()
    
    # Ask for confirmation
    print(f"\nThis experiment will run {len(experiment.windows)} windows.")
    print(f"Estimated runtime: {len(experiment.windows) * 20:.0f}-{len(experiment.windows) * 30:.0f} minutes")
    
    response = input("\nProceed with experiment? [y/N]: ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return
    
    # Run DQN experiment
    print(f"\n=== Starting DQN Sliding Window Experiment ===")
    dqn_results = experiment.run_full_experiment(algorithm='dqn')
    
    # Optionally run TD3 experiment
    run_td3 = input("\nRun TD3 experiment as well? [y/N]: ")
    if run_td3.lower() == 'y':
        print(f"\n=== Starting TD3 Sliding Window Experiment ===")
        experiment.base_exp_id = "sliding_window_td3"  # Different base ID
        td3_results = experiment.run_full_experiment(algorithm='td3')
    
    print("\n=== All Experiments Complete ===")
    
    return experiment


if __name__ == "__main__":
    experiment = main()