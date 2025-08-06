#!/usr/bin/env python3
"""
Ablation Experiment: Grid Search across Trading Frequency and Years

This experiment systematically tests:
- Trading frequencies: 1, 5, 10, 20 days
- Years: 2023, 2024, 2025 (by adjusting base_offset)
- Algorithm: DQN with 2 episodes for speed
- Transaction costs: 0.15% (back to original)

Grid: 4 frequencies × 3 years = 12 total configurations
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
from itertools import product

# Add paths
sys.path.append('/home/ubuntu/code/angle_rl/invest')

# Import utilities and training functions
from utils import aggregate_tickers_RL
from train_with_dqn import train_financial_dqn, evaluate_financial_dqn
from financial_dqn_agent import create_financial_dqn_agent


class AblationGridSearchExperiment:
    """Grid search experiment across trading frequency and years."""
    
    def __init__(self, 
                 data_list_file: str = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt",
                 base_exp_id: str = "ablation_grid_search",
                 training_window_size: int = 265,
                 eval_window_size: int = 60,
                 device: str = None,
                 algorithm: str = "dqn"):
        
        self.data_list_file = data_list_file
        self.base_exp_id = base_exp_id
        self.training_window_size = training_window_size
        self.eval_window_size = eval_window_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.algorithm = algorithm
        
        # Grid search parameters
        self.trading_frequencies = [1, 5, 10, 20]  # days
        self.year_configs = {
            '2023': 250,   # base_offset for ~2023 period
            '2024': 500,   # base_offset for ~2024 period  
            '2025': 750    # base_offset for ~2025 period
        }
        
        # Check data availability
        with open(data_list_file, 'r') as f:
            self.total_files = len([line.strip() for line in f.readlines() if line.strip()])
        
        print(f"Total data files available: {self.total_files}")
        
        # Results storage
        self.results = {
            'experiment_config': {
                'training_window_size': training_window_size,
                'eval_window_size': eval_window_size,
                'trading_frequencies': self.trading_frequencies,
                'year_configs': self.year_configs,
                'total_files_available': self.total_files,
                'transaction_cost_ratio': 0.0015,  # Back to 0.15%
                'algorithm': self.algorithm,
                'episodes_per_run': 2,
                'device': self.device,
                'description': f'Ablation study: trading frequency vs years grid search ({self.algorithm})'
            },
            'grid_results': [],
            'summary': {}
        }
    
    def calculate_window_for_config(self, year_config: str, base_offset: int) -> Dict:
        """Calculate single window specification for a year configuration."""
        train_start = base_offset
        train_end = train_start + self.training_window_size
        eval_start = train_end
        eval_end = eval_start + self.eval_window_size
        
        # Check if we have enough data
        if eval_end > self.total_files:
            raise ValueError(f"Year {year_config} would need {eval_end} files but only {self.total_files} available")
        
        return {
            'year_config': year_config,
            'train_start': train_start,
            'train_end': train_end,
            'eval_start': eval_start,
            'eval_end': eval_end,
            'train_days': train_end - train_start,
            'eval_days': eval_end - eval_start
        }
    
    def show_date_mapping_for_config(self, window_spec: Dict):
        """Display actual dates for a configuration."""
        # Read data files
        with open(self.data_list_file, 'r') as f:
            data_files = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"  Year {window_spec['year_config']} date mapping:")
        
        # Extract dates from filenames
        if window_spec['train_start'] < len(data_files):
            train_start_file = data_files[window_spec['train_start']]
            train_start_date = train_start_file.split('training_data_start_date_')[1].split('_test')[0].replace('_', '-')
            print(f"    Training start: {train_start_date}")
        
        if window_spec['eval_end']-1 < len(data_files):
            eval_end_file = data_files[window_spec['eval_end']-1]
            eval_end_date = eval_end_file.split('training_data_start_date_')[1].split('_test')[0].replace('_', '-')
            print(f"    Evaluation end: {eval_end_date}")
    
    def create_ticker_hash_for_config(self, window_spec: Dict, trading_freq: int) -> str:
        """Create ticker hash for a specific configuration."""
        exp_id = f"{self.base_exp_id}_{window_spec['year_config']}_freq{trading_freq}d"
        ticker_hash_file = f"/home/ubuntu/code/angle_rl/invest/{exp_id}_ticker_hash.pkl"
        
        # Skip if already exists
        if os.path.exists(ticker_hash_file):
            print(f"    Reusing existing ticker hash: {ticker_hash_file}")
            return ticker_hash_file
        
        # Read data files
        with open(self.data_list_file, 'r') as f:
            data_files = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"    Creating ticker hash for {window_spec['year_config']} (files {window_spec['train_start']}-{window_spec['train_end']})...")
        
        # Create ticker hash for this configuration's training period
        aggregate_tickers_RL(
            data_file_list=data_files,
            start_idx=window_spec['train_start'],
            end_idx_plus1=window_spec['train_end'],
            exp_id=exp_id
        )
        
        return ticker_hash_file
    
    def run_single_configuration(self, year_config: str, base_offset: int, trading_freq: int, 
                                num_episodes: int = 2) -> Dict:
        """Run experiment for a single configuration."""
        config_id = f"{year_config}_freq{trading_freq}d"
        print(f"\n=== Configuration: {config_id} ===")
        print(f"Year: {year_config}, Trading frequency: {trading_freq} days, Episodes: {num_episodes}")
        
        start_time = time.time()
        
        try:
            # Calculate window
            window_spec = self.calculate_window_for_config(year_config, base_offset)
            self.show_date_mapping_for_config(window_spec)
            
            # Create/get ticker hash
            ticker_hash_file = self.create_ticker_hash_for_config(window_spec, trading_freq)
            
            # Load ticker data
            with open(ticker_hash_file, 'rb') as f:
                ticker_data = pickle.load(f)
            
            print(f"  Ticker count: {ticker_data['num_tickers']}")
            
            # Create experiment ID
            exp_id = f"{self.base_exp_id}_{config_id}"
            
            # Training parameters - KEY: trading frequency variation
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
                'action_update_interval': trading_freq,  # 🎯 KEY VARIABLE: Trading frequency
                'transaction_cost_ratio': 0.0015,  # Back to 0.15%
                # Training parameters  
                'log_interval': 1,  # More frequent logging for short runs
                'save_interval': 10,
                'device': self.device,
                'seed': 42,  # Fixed seed for reproducibility
                # TD3 parameters (used if algorithm is 'td3')
                'use_twin_networks': self.algorithm == 'td3',
                'use_tau_updates': self.algorithm == 'td3',
                'tau': 0.005,  # TD3 soft update rate
                'policy_delay': 2,  # TD3 delayed policy updates
                'target_noise': 0.2,  # TD3 target policy smoothing noise
                'noise_clip': 0.5    # TD3 noise clipping
            }
            
            print(f"  💰 Transaction cost: {training_params['transaction_cost_ratio']:.2%}")
            print(f"  🔄 Trading frequency: every {trading_freq} days")
            print(f"  🤖 Algorithm: {self.algorithm.upper()} {'(TD3 features enabled)' if self.algorithm == 'td3' else ''}")
            
            # Train the agent
            print(f"  Starting training ({num_episodes} episodes)...")
            train_start_time = time.time()
            
            # Log TD3 features if enabled
            if self.algorithm == 'td3':
                print(f"    🎯 TD3 features: twin_networks={training_params['use_twin_networks']}, tau_updates={training_params['use_tau_updates']}, policy_delay={training_params['policy_delay']}")
            
            # Filter out unsupported TD3 parameters
            train_params = {k: v for k, v in training_params.items() 
                          if k not in ['target_noise', 'noise_clip']}
            
            agent, history = train_financial_dqn(**train_params)
            
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
                action_update_interval=trading_freq,  # Same frequency for evaluation
                transaction_cost_ratio=0.0015,  # Same transaction cost
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
            config_results = {
                'configuration': {
                    'year_config': year_config,
                    'trading_frequency_days': trading_freq,
                    'base_offset': base_offset,
                    'config_id': config_id
                },
                'window_spec': window_spec,
                'algorithm': self.algorithm,
                'transaction_cost_ratio': 0.0015,
                'training': {
                    'episodes': num_episodes,
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
            
            print(f"  ✅ Results: Train={train_return:.2f}%, Eval={eval_return:.2f}%, Sharpe={eval_results['avg_sharpe']:.4f}")
            print(f"  ⏱️ Total time: {total_time:.1f}s")
            
            return config_results
            
        except Exception as e:
            print(f"  ❌ ERROR in configuration {config_id}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'configuration': {
                    'year_config': year_config,
                    'trading_frequency_days': trading_freq,
                    'base_offset': base_offset,
                    'config_id': config_id
                },
                'error': str(e),
                'algorithm': self.algorithm,
                'transaction_cost_ratio': 0.0015
            }
    
    def run_full_grid_search(self, num_episodes: int = 2):
        """Run the complete grid search experiment."""
        total_configs = len(self.trading_frequencies) * len(self.year_configs)
        
        print(f"\n{'='*70}")
        print(f"ABLATION GRID SEARCH EXPERIMENT")
        print(f"{'='*70}")
        print(f"Trading frequencies: {self.trading_frequencies} days")
        print(f"Years: {list(self.year_configs.keys())}")
        print(f"Total configurations: {total_configs}")
        print(f"Episodes per config: {num_episodes}")
        print(f"Transaction costs: 0.15% (restored)")
        print(f"Device: {self.device}")
        print(f"{'='*70}")
        
        experiment_start_time = time.time()
        config_count = 0
        
        # Run grid search
        for year_config, base_offset in self.year_configs.items():
            for trading_freq in self.trading_frequencies:
                config_count += 1
                print(f"\n🔬 Configuration {config_count}/{total_configs}")
                
                config_results = self.run_single_configuration(
                    year_config=year_config,
                    base_offset=base_offset,
                    trading_freq=trading_freq,
                    num_episodes=num_episodes
                )
                
                self.results['grid_results'].append(config_results)
        
        # Calculate summary statistics
        successful_configs = [r for r in self.results['grid_results'] if 'error' not in r]
        
        if successful_configs:
            eval_returns = [r['evaluation']['return_pct'] for r in successful_configs]
            sharpe_ratios = [r['evaluation']['sharpe_ratio'] for r in successful_configs]
            
            # Group by trading frequency
            freq_stats = {}
            for freq in self.trading_frequencies:
                freq_configs = [r for r in successful_configs 
                              if r['configuration']['trading_frequency_days'] == freq]
                if freq_configs:
                    freq_returns = [r['evaluation']['return_pct'] for r in freq_configs]
                    freq_stats[freq] = {
                        'count': len(freq_configs),
                        'avg_return': np.mean(freq_returns),
                        'std_return': np.std(freq_returns),
                        'min_return': np.min(freq_returns),
                        'max_return': np.max(freq_returns)
                    }
            
            # Group by year
            year_stats = {}
            for year in self.year_configs.keys():
                year_configs = [r for r in successful_configs 
                              if r['configuration']['year_config'] == year]
                if year_configs:
                    year_returns = [r['evaluation']['return_pct'] for r in year_configs]
                    year_stats[year] = {
                        'count': len(year_configs),
                        'avg_return': np.mean(year_returns),
                        'std_return': np.std(year_returns),
                        'min_return': np.min(year_returns),
                        'max_return': np.max(year_returns)
                    }
            
            self.results['summary'] = {
                'total_configs': total_configs,
                'successful_configs': len(successful_configs),
                'failed_configs': total_configs - len(successful_configs),
                'overall_stats': {
                    'avg_eval_return': np.mean(eval_returns),
                    'std_eval_return': np.std(eval_returns),
                    'min_eval_return': np.min(eval_returns),
                    'max_eval_return': np.max(eval_returns),
                    'avg_sharpe_ratio': np.mean(sharpe_ratios)
                },
                'trading_frequency_stats': freq_stats,
                'year_stats': year_stats,
                'total_experiment_time': time.time() - experiment_start_time
            }
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{self.base_exp_id}_{self.algorithm}_{num_episodes}ep_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.print_summary()
        print(f"\n📊 Results saved to: {results_file}")
        
        return self.results
    
    def print_summary(self):
        """Print experiment summary."""
        print(f"\n{'='*70}")
        print(f"GRID SEARCH EXPERIMENT COMPLETED")
        print(f"{'='*70}")
        
        if 'summary' in self.results and self.results['summary']:
            summary = self.results['summary']
            
            print(f"\n📈 Overall Performance:")
            print(f"  Successful configs: {summary['successful_configs']}/{summary['total_configs']}")
            print(f"  Average return: {summary['overall_stats']['avg_eval_return']:.2f}%")
            print(f"  Return std: {summary['overall_stats']['std_eval_return']:.2f}%")
            print(f"  Return range: {summary['overall_stats']['min_eval_return']:.2f}% to {summary['overall_stats']['max_eval_return']:.2f}%")
            print(f"  Average Sharpe: {summary['overall_stats']['avg_sharpe_ratio']:.4f}")
            
            print(f"\n🔄 Trading Frequency Analysis:")
            for freq, stats in summary['trading_frequency_stats'].items():
                print(f"  {freq}d frequency: {stats['avg_return']:+.2f}% ± {stats['std_return']:.2f}% (n={stats['count']})")
            
            print(f"\n📅 Year Analysis:")
            for year, stats in summary['year_stats'].items():
                print(f"  {year}: {stats['avg_return']:+.2f}% ± {stats['std_return']:.2f}% (n={stats['count']})")
            
            print(f"\n⏱️ Total experiment time: {summary['total_experiment_time']:.1f}s")


def main():
    """Run the ablation grid search experiment."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation grid search experiment')
    parser.add_argument('--episodes', type=int, default=2,
                        help='Number of episodes per configuration')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--algorithm', type=str, default='dqn', choices=['dqn', 'td3'],
                        help='Algorithm to use: dqn (standard) or td3 (with TD3 features)')
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = AblationGridSearchExperiment(device=args.device, algorithm=args.algorithm)
    results = experiment.run_full_grid_search(num_episodes=args.episodes)
    
    return results


if __name__ == "__main__":
    main()