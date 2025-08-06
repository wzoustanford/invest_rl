#!/usr/bin/env python3
"""
Sliding Window Experiment for Financial DQN/TD3 - 2023 Period
Test multiple time windows to get ~240 evaluation days (~1 year of results)

This version shifts the experiment forward by approximately 1 year (250 trading days)
to avoid the 2022 bear market and test on 2023 data instead.

Original Design (2021-2022):
- Window 1: Train[0:265], Eval[265:325] (March 2021 - June 2021)
- Window 2: Train[60:325], Eval[325:385] (June 2021 - September 2021)
- Window 3: Train[120:385], Eval[385:445] (September 2021 - December 2021)
- Window 4: Train[180:445], Eval[445:505] (December 2021 - February 2022)

New Design (2022-2023):
- Window 1: Train[250:515], Eval[515:575] (March 2022 - June 2022)
- Window 2: Train[310:575], Eval[575:635] (June 2022 - September 2022)
- Window 3: Train[370:635], Eval[635:695] (September 2022 - December 2022)
- Window 4: Train[430:695], Eval[695:755] (December 2022 - March 2023)

Total: 240 evaluation days across 4 windows in 2022-2023
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


class SlidingWindow2023Experiment:
    """Manages sliding window experiments for financial RL - 2023 version."""
    
    def __init__(self, 
                 data_list_file: str = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt",
                 base_exp_id: str = "sliding_window_2023",
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
        self.base_offset = base_offset  # New: offset all windows by this amount
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
                'description': 'Sliding window experiment shifted to 2023 period'
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
        print("\n=== Date Mapping for 2023 Experiment Windows ===\n")
        
        # Read all data files
        with open(self.data_list_file, 'r') as f:
            data_files = [line.strip() for line in f.readlines() if line.strip()]
        
        for window in self.windows:
            print(f"Window {window['window_id']}:")
            print(f"  Training: indices {window['train_start']}-{window['train_end']}")
            
            # Extract dates from filenames
            if window['train_start'] < len(data_files):
                train_start_file = data_files[window['train_start']]
                # Extract date from filename
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
    
    def run_window_experiment(self, window_spec: Dict, algorithm: str = 'dqn', num_episodes: int = 10) -> Dict:
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
            'transaction_cost_ratio': 0.0015,
            # Training parameters  
            'log_interval': 2,
            'save_interval': 10,
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
            print(f"  ERROR in window {window_spec['window_id']}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'window_spec': window_spec,
                'error': str(e),
                'algorithm': algorithm
            }
    
    def run_full_experiment(self, algorithm: str = 'dqn', num_episodes: int = 10):
        """Run the full sliding window experiment."""
        print(f"\n{'='*60}")
        print(f"SLIDING WINDOW EXPERIMENT - 2023 PERIOD")
        print(f"{'='*60}")
        print(f"Algorithm: {algorithm.upper()}")
        print(f"Episodes per window: {num_episodes}")
        print(f"Windows: {len(self.windows)}")
        print(f"Total evaluation days: {sum(w['eval_days'] for w in self.windows)}")
        print(f"Device: {self.device}")
        print(f"Base offset: {self.base_offset} days (shifting to 2023 period)")
        
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
        print(f"EXPERIMENT COMPLETED")
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
    """Run the 2023 sliding window experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run sliding window experiment for 2023 period')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'td3'],
                        help='Algorithm to use (dqn or td3)')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of episodes per window')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = SlidingWindow2023Experiment(device=args.device)
    results = experiment.run_full_experiment(algorithm=args.algorithm, num_episodes=args.episodes)
    
    return results


if __name__ == "__main__":
    main()