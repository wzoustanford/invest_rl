#!/usr/bin/env python3
"""
Sliding Window Experiment for Financial DQN/TD3 - 2023 Period with ZERO Trading Costs
Same as the original 2023 experiment but with transaction_cost_ratio = 0.0

This will help us understand the impact of trading costs on performance.
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


class SlidingWindow2023ZeroCostExperiment:
    """Manages sliding window experiments for financial RL - 2023 version with zero trading costs."""
    
    def __init__(self, 
                 data_list_file: str = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt",
                 base_exp_id: str = "sliding_window_2023_zero_cost",
                 training_window_size: int = 265,
                 eval_window_size: int = 60,
                 window_shift: int = 60,
                 base_offset: int = 250,  # Shift forward by ~1 year (250 trading days)
                 target_eval_days: int = 240,
                 device: str = None):
        
        self.data_list_file = data_list_file
        self.base_exp_id = base_exp_id
        self.training_window_size = training_window_size
        self.eval_window_size = eval_window_size
        self.window_shift = window_shift
        self.base_offset = base_offset
        self.target_eval_days = target_eval_days
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Calculate number of windows needed
        self.num_windows = target_eval_days // eval_window_size
        
        # Check data availability
        with open(data_list_file, 'r') as f:
            self.total_files = len([line.strip() for line in f.readlines() if line.strip()])
        
        print(f"Total data files available: {self.total_files}")
        
        # Calculate window specifications
        self.windows = self._calculate_windows()
        
        # Results storage
        self.results = {
            'experiment_config': {
                'training_window_size': training_window_size,
                'eval_window_size': eval_window_size,
                'window_shift': window_shift,
                'base_offset': base_offset,
                'num_windows': self.num_windows,
                'target_eval_days': target_eval_days,
                'device': self.device,
                'total_files_available': self.total_files,
                'transaction_cost_ratio': 0.0,  # Key difference: ZERO trading costs
                'description': 'Sliding window experiment shifted to 2023 period with ZERO trading costs'
            },
            'windows': [],
            'summary': {}
        }
    
    def _calculate_windows(self) -> List[Dict]:
        """Calculate window specifications with base offset."""
        windows = []
        
        for i in range(self.num_windows):
            # Each window shifts by window_shift, plus base offset
            train_start = self.base_offset + (i * self.window_shift)
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
    
    def show_date_mapping(self):
        """Display the actual dates for each window based on file indices."""
        print("\n=== Date Mapping for 2023 Zero-Cost Experiment Windows ===\n")
        
        # Read all data files
        with open(self.data_list_file, 'r') as f:
            data_files = [line.strip() for line in f.readlines() if line.strip()]
        
        for window in self.windows:
            print(f"Window {window['window_id']}:")
            print(f"  Training: indices {window['train_start']}-{window['train_end']}")
            
            # Extract dates from filenames
            if window['train_start'] < len(data_files):
                train_start_file = data_files[window['train_start']]
                train_start_date = train_start_file.split('training_data_start_date_')[1].split('_test')[0].replace('_', '-')
                print(f"    Start: {train_start_date}")
            
            if window['train_end']-1 < len(data_files):
                train_end_file = data_files[window['train_end']-1]
                train_end_date = train_end_file.split('training_data_start_date_')[1].split('_test')[0].replace('_', '-')
                print(f"    End:   {train_end_date}")
            
            print(f"  Evaluation: indices {window['eval_start']}-{window['eval_end']}")
            
            if window['eval_start'] < len(data_files):
                eval_start_file = data_files[window['eval_start']]
                eval_start_date = eval_start_file.split('training_data_start_date_')[1].split('_test')[0].replace('_', '-')
                print(f"    Start: {eval_start_date}")
            
            if window['eval_end']-1 < len(data_files):
                eval_end_file = data_files[window['eval_end']-1]
                eval_end_date = eval_end_file.split('training_data_start_date_')[1].split('_test')[0].replace('_', '-')
                print(f"    End:   {eval_end_date}")
            
            print()
    
    def run_window_experiment(self, window_spec: Dict, algorithm: str = 'dqn', num_episodes: int = 10) -> Dict:
        """Run experiment for a single window with zero trading costs."""
        print(f"\n=== Window {window_spec['window_id']} Experiment (ZERO TRADING COSTS) ===")
        print(f"Training: files {window_spec['train_start']}-{window_spec['train_end']} ({window_spec['train_days']} days)")
        print(f"Evaluation: files {window_spec['eval_start']}-{window_spec['eval_end']} ({window_spec['eval_days']} days)")
        
        start_time = time.time()
        
        # Reuse ticker hash from original experiment if it exists
        original_ticker_hash_file = f"/home/ubuntu/code/angle_rl/invest/sliding_window_2023_w{window_spec['window_id']}_ticker_hash.pkl"
        
        if os.path.exists(original_ticker_hash_file):
            ticker_hash_file = original_ticker_hash_file
            print(f"  Reusing ticker hash from original experiment: {ticker_hash_file}")
        else:
            # Create new ticker hash
            ticker_hash_file = self.create_ticker_hash(window_spec)
        
        # Load ticker data
        with open(ticker_hash_file, 'rb') as f:
            ticker_data = pickle.load(f)
        
        print(f"  Ticker hash: {ticker_data['num_tickers']} unique tickers")
        
        # Create agent
        exp_id = f"{self.base_exp_id}_w{window_spec['window_id']}"
        
        print(f"  Algorithm: {algorithm.upper()}")
        print(f"  🆓 ZERO TRADING COSTS (vs 0.15% in original)")
        
        # Training parameters - KEY CHANGE: transaction_cost_ratio = 0.0
        training_params = {
            'data_list_filename': self.data_list_file,
            'ticker_hash_file': ticker_hash_file,
            'exp_id': exp_id,
            'start_date_idx': window_spec['train_start'],
            'end_date_idx_plus1': window_spec['train_end'],
            'eval_start_date_idx': window_spec['eval_start'],
            'eval_end_date_idx_plus1': window_spec['eval_end'],
            # DQN parameters
            'num_episodes': num_episodes,
            'num_discrete_actions': 200,
            'gamma': 0.99,
            'lr': 0.0001,
            'batch_size': 64,
            'memory_size': 20000,
            'epsilon_start': 0.3,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_frequency': 300,
            # Environment parameters
            'action_update_interval': 10,
            'transaction_cost_ratio': 0.0,  # 🎯 KEY CHANGE: ZERO trading costs
            # Training parameters  
            'log_interval': 2,
            'save_interval': 10,
            'device': self.device,
            'seed': 42 + window_spec['window_id']  # Different seed per window
        }
        
        print(f"  Training parameters: episodes={training_params['num_episodes']}, gamma={training_params['gamma']}, lr={training_params['lr']}")
        print(f"  💰 Transaction cost: {training_params['transaction_cost_ratio']:.1%} (FREE!)")
        
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
                transaction_cost_ratio=0.0,  # 🎯 Also zero for evaluation
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
                'transaction_cost_ratio': 0.0,  # Record the zero cost
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
            print(f"  ERROR in window {window_spec['window_id']}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'window_spec': window_spec,
                'error': str(e),
                'algorithm': algorithm,
                'transaction_cost_ratio': 0.0
            }
    
    def create_ticker_hash(self, window_spec: Dict) -> str:
        """Create ticker hash for a specific window (fallback if not reusing)."""
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
    
    def run_full_experiment(self, algorithm: str = 'dqn', num_episodes: int = 10):
        """Run the full sliding window experiment with zero trading costs."""
        print(f"\n{'='*60}")
        print(f"SLIDING WINDOW EXPERIMENT - 2023 PERIOD - ZERO TRADING COSTS")
        print(f"{'='*60}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Episodes per window: {num_episodes}")
        print(f"Windows: {len(self.windows)}")
        print(f"Total evaluation days: {sum(w['eval_days'] for w in self.windows)}")
        print(f"Device: {self.device}")
        print(f"Base offset: {self.base_offset} days (shifting to 2023 period)")
        print(f"💰 Transaction costs: 0.0% (FREE TRADING!)")
        
        # Show date mapping
        self.show_date_mapping()
        
        experiment_start_time = time.time()
        
        # Run each window
        for window in self.windows:
            window_results = self.run_window_experiment(window, algorithm=algorithm, num_episodes=num_episodes)
            self.results['windows'].append(window_results)
        
        # Calculate summary statistics
        successful_windows = [w for w in self.results['windows'] if 'error' not in w]
        
        if successful_windows:
            eval_returns = [w['evaluation']['return_pct'] for w in successful_windows]
            sharpe_ratios = [w['evaluation']['sharpe_ratio'] for w in successful_windows]
            
            self.results['summary'] = {
                'algorithm': algorithm,
                'num_episodes': num_episodes,
                'transaction_cost_ratio': 0.0,
                'successful_windows': len(successful_windows),
                'failed_windows': len(self.results['windows']) - len(successful_windows),
                'avg_eval_return': np.mean(eval_returns),
                'std_eval_return': np.std(eval_returns),
                'min_eval_return': np.min(eval_returns),
                'max_eval_return': np.max(eval_returns),
                'avg_sharpe_ratio': np.mean(sharpe_ratios),
                'total_experiment_time': time.time() - experiment_start_time
            }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.base_exp_id}_{algorithm}_{num_episodes}ep_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n{'='*60}")
        print(f"ZERO-COST EXPERIMENT COMPLETED")
        print(f"{'='*60}")
        
        if successful_windows:
            print(f"Summary Statistics:")
            print(f"  Successful windows: {self.results['summary']['successful_windows']}/{len(self.windows)}")
            print(f"  Average eval return: {self.results['summary']['avg_eval_return']:.2f}%")
            print(f"  Std eval return: {self.results['summary']['std_eval_return']:.2f}%")
            print(f"  Min/Max eval return: {self.results['summary']['min_eval_return']:.2f}% / {self.results['summary']['max_eval_return']:.2f}%")
            print(f"  Average Sharpe ratio: {self.results['summary']['avg_sharpe_ratio']:.4f}")
            print(f"  Total time: {self.results['summary']['total_experiment_time']:.1f}s")
        
        print(f"\nResults saved to: {results_file}")
        
        return self.results


def main():
    """Run the 2023 sliding window experiment with zero trading costs."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sliding window experiment for 2023 period with zero trading costs')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'td3'],
                        help='Algorithm to use (dqn or td3)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes per window')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = SlidingWindow2023ZeroCostExperiment(device=args.device)
    results = experiment.run_full_experiment(algorithm=args.algorithm, num_episodes=args.episodes)
    
    return results


if __name__ == "__main__":
    main()