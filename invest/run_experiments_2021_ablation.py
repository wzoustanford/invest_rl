"""
Comprehensive Experiment Runner for 2021 + Ablation Studies
Tests Sequential Supervised, DQN, and TD3 with various configurations
"""

import torch
import pickle
import numpy as np
import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple
import gc

from model.iimodel import IIMODEL
from financial_dqn_agent import create_financial_dqn_agent
from financial_env import FinancialTradingEnv
from train_with_dqn import train_financial_dqn


class ComprehensiveExperimentRunner:
    """Run all experiments with different configurations."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/ablation_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all data files
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files")
        print(f"Output directory: {self.output_dir}")
        
        # Results storage
        self.results = {
            'experiments': [],
            'summary': {}
        }
    
    def extract_date_from_filename(self, filename: str) -> Tuple[str, str]:
        """Extract training and test dates from filename."""
        pattern = r'training_data_start_date_(\d{4}_\d{2}_\d{2})_test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            train_date = match.group(1).replace('_', '-')
            test_date = match.group(2).replace('_', '-')
            return train_date, test_date
        return None, None
    
    def find_month_file(self, year: int, month: int) -> Tuple[int, str]:
        """Find file for specific month."""
        for i, filepath in enumerate(self.all_files):
            _, test_date = self.extract_date_from_filename(filepath)
            if test_date:
                file_year = int(test_date[:4])
                file_month = int(test_date[5:7])
                if file_year == year and file_month == month:
                    return i, filepath
        return None, None
    
    # ========== SEQUENTIAL SUPERVISED LEARNING ==========
    
    def train_sequential_model(self, 
                              training_files: List[str],
                              gamma: float = 0.3,
                              iterations: int = 100,
                              sequence_days: int = 7) -> torch.nn.Module:
        """Train Sequential Supervised model with specified parameters."""
        
        # Load sequence of data files
        data_sequence = []
        for i, filepath in enumerate(training_files[:sequence_days]):
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data_sequence.append(pickle.load(f))
        
        if len(data_sequence) < sequence_days:
            return None
        
        # Initialize model
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        for step in range(iterations):
            model.train()
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0).to(self.device)
            
            for i in range(len(data_sequence)):
                features = data_sequence[i]['trainFeature'].to(self.device)
                series = data_sequence[i]['train_in_portfolio_series'].to(self.device)
                
                # Get portfolio weights
                weights = model(features)
                
                # Calculate return and Sharpe
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                # Apply gamma discounting
                gamma_power = gamma ** (len(data_sequence) - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            if (step + 1) % 100 == 0:
                print(f"      Step {step+1}/{iterations}, Loss: {total_loss.item():.4f}", end='\r')
        
        return model
    
    def evaluate_sequential_model(self, model: torch.nn.Module, test_file: str) -> Dict:
        """Evaluate Sequential model."""
        if not os.path.exists(test_file) or model is None:
            return None
        
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        if test_data.get('test_in_portfolio_series') is None:
            return None
        
        model.eval()
        with torch.no_grad():
            features = test_data['testFeature'].to(self.device)
            series = test_data['test_in_portfolio_series'].to(self.device)
            
            weights = model(features)
            shares = weights / (series[:, 0:1] + 1e-10)
            ret = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            
            # Apply transaction cost
            ret = ret - 0.0015
            
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = ret / (torch.std(daily_returns) + 1e-10)
            
            return {
                'return': ret.item(),
                'sharpe': sharpe.item()
            }
    
    def run_sequential_year(self, 
                           year: int,
                           gamma: float = 0.3,
                           iterations: int = 100,
                           sequence_days: int = 7) -> Dict:
        """Run Sequential Supervised for a full year."""
        
        print(f"\n  Sequential Supervised - Year {year} (gamma={gamma}, iter={iterations}, seq={sequence_days})")
        
        year_results = {
            'algorithm': 'Sequential',
            'year': year,
            'gamma': gamma,
            'iterations': iterations,
            'sequence_days': sequence_days,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < sequence_days - 1:
                continue
            
            # Get training files
            training_files = self.all_files[file_idx - sequence_days + 1:file_idx + 1]
            
            # Train model
            model = self.train_sequential_model(training_files, gamma, iterations, sequence_days)
            
            # Evaluate
            result = self.evaluate_sequential_model(model, test_file)
            
            if result:
                year_results['cumulative'] *= (1 + result['return'])
                year_results['monthly_trades'].append({
                    'month': month,
                    'return': result['return'],
                    'sharpe': result['sharpe']
                })
                
                print(f"    Month {month:02d}: {result['return']*100:+.2f}%", end=' ')
                if month % 4 == 0:
                    print()
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
        
        # Calculate statistics
        if year_results['monthly_trades']:
            returns = [t['return'] for t in year_results['monthly_trades']]
            year_results['annual_return'] = year_results['cumulative'] - 1
            year_results['avg_return'] = np.mean(returns)
            year_results['num_trades'] = len(returns)
            year_results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
        
        print(f"\n    Annual: {(year_results['cumulative']-1)*100:+.2f}%, Trades: {len(year_results['monthly_trades'])}")
        
        return year_results
    
    # ========== DQN ALGORITHMS ==========
    
    def run_dqn_year(self, year: int, use_td3: bool = False) -> Dict:
        """Run DQN or TD3 for a year."""
        
        algo_name = "TD3" if use_td3 else "DQN"
        print(f"\n  {algo_name} - Year {year}")
        
        # Find appropriate data window for the year
        # Use Q2 data (April-June) as representative
        april_idx, _ = self.find_month_file(year, 4)
        
        if april_idx is None or april_idx < 265:
            print(f"    Insufficient data for {year}")
            return None
        
        # Setup training window (265 files) and eval window (60 files)
        train_start = april_idx - 265
        train_end = april_idx
        eval_start = april_idx
        eval_end = min(april_idx + 60, len(self.all_files))
        
        # Create environment
        env_config = {
            'data_dir': self.data_dir,
            'tickers': [],  # Will be loaded from files
            'features': [],  # Will be loaded from files
            'start_date': None,
            'end_date': None,
            'trading_days': None,
            'transaction_cost_rate': 0.0015,
            'data_files': self.all_files[train_start:train_end]
        }
        
        try:
            # Train agent
            results = train_financial_dqn(
                data_dir=self.data_dir,
                file_list=self.all_files[train_start:train_end],
                eval_file_list=self.all_files[eval_start:eval_end],
                episodes=2,
                use_twin_networks=use_td3,
                use_tau_updates=use_td3,
                tau=0.005 if use_td3 else 1.0,
                policy_delay=2 if use_td3 else 1,
                device=self.device
            )
            
            year_results = {
                'algorithm': algo_name,
                'year': year,
                'annual_return': results.get('eval_return', 0),
                'sharpe': results.get('eval_sharpe', 0),
                'train_window': f"files {train_start}-{train_end}",
                'eval_window': f"files {eval_start}-{eval_end}"
            }
            
            print(f"    Return: {results.get('eval_return', 0)*100:+.2f}%")
            
        except Exception as e:
            print(f"    Error: {str(e)}")
            year_results = {
                'algorithm': algo_name,
                'year': year,
                'error': str(e)
            }
        
        return year_results
    
    # ========== MAIN EXPERIMENT RUNNER ==========
    
    def run_all_experiments(self):
        """Run all planned experiments."""
        
        print("\n" + "="*80)
        print("COMPREHENSIVE EXPERIMENT RUNNER")
        print("="*80)
        
        experiments_config = [
            # Baseline experiments
            {'algo': 'Sequential', 'years': [2021, 2022, 2023], 'gamma': 0.3, 'iter': 100, 'seq': 7},
            {'algo': 'DQN', 'years': [2021, 2022, 2023]},
            {'algo': 'TD3', 'years': [2021, 2022, 2023]},
            
            # Ablation: 750 iterations
            {'algo': 'Sequential', 'years': [2021, 2022, 2023], 'gamma': 0.3, 'iter': 750, 'seq': 7},
            
            # Ablation: 14-day sequences
            {'algo': 'Sequential', 'years': [2021, 2022, 2023], 'gamma': 0.3, 'iter': 100, 'seq': 14},
            
            # Ablation: Gamma values
            {'algo': 'Sequential', 'years': [2021, 2022, 2023], 'gamma': 0.1, 'iter': 100, 'seq': 7},
            {'algo': 'Sequential', 'years': [2021, 2022, 2023], 'gamma': 0.5, 'iter': 100, 'seq': 7},
        ]
        
        for config in experiments_config:
            algo = config['algo']
            
            if algo == 'Sequential':
                print(f"\n\n>>> Sequential Supervised (gamma={config['gamma']}, iter={config['iter']}, seq={config['seq']})")
                for year in config['years']:
                    result = self.run_sequential_year(
                        year, 
                        gamma=config['gamma'],
                        iterations=config['iter'],
                        sequence_days=config['seq']
                    )
                    self.results['experiments'].append(result)
                    
            elif algo == 'DQN':
                print(f"\n\n>>> DQN Standard")
                for year in config['years']:
                    result = self.run_dqn_year(year, use_td3=False)
                    if result:
                        self.results['experiments'].append(result)
                        
            elif algo == 'TD3':
                print(f"\n\n>>> TD3 (Twin Q-Networks)")
                for year in config['years']:
                    result = self.run_dqn_year(year, use_td3=True)
                    if result:
                        self.results['experiments'].append(result)
            
            # Save intermediate results
            self.save_results()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def generate_summary(self):
        """Generate summary statistics."""
        
        # Group results by configuration
        baseline_seq = [r for r in self.results['experiments'] 
                        if r.get('algorithm') == 'Sequential' and r.get('iterations') == 100 and r.get('sequence_days') == 7 and r.get('gamma') == 0.3]
        
        iter750 = [r for r in self.results['experiments']
                  if r.get('algorithm') == 'Sequential' and r.get('iterations') == 750]
        
        seq14 = [r for r in self.results['experiments']
                if r.get('algorithm') == 'Sequential' and r.get('sequence_days') == 14]
        
        gamma01 = [r for r in self.results['experiments']
                  if r.get('algorithm') == 'Sequential' and r.get('gamma') == 0.1]
        
        gamma05 = [r for r in self.results['experiments']
                  if r.get('algorithm') == 'Sequential' and r.get('gamma') == 0.5]
        
        dqn_results = [r for r in self.results['experiments'] if r.get('algorithm') == 'DQN']
        td3_results = [r for r in self.results['experiments'] if r.get('algorithm') == 'TD3']
        
        # Calculate average performance
        def avg_return(results):
            returns = [r.get('annual_return', 0) for r in results if 'annual_return' in r]
            return np.mean(returns) if returns else 0
        
        self.results['summary'] = {
            'baseline_sequential': {
                'avg_annual_return': avg_return(baseline_seq),
                'results_by_year': {r['year']: r.get('annual_return', 0) for r in baseline_seq}
            },
            'iterations_750': {
                'avg_annual_return': avg_return(iter750),
                'improvement_vs_baseline': avg_return(iter750) - avg_return(baseline_seq)
            },
            'sequence_14days': {
                'avg_annual_return': avg_return(seq14),
                'improvement_vs_baseline': avg_return(seq14) - avg_return(baseline_seq)
            },
            'gamma_0.1': {
                'avg_annual_return': avg_return(gamma01),
                'improvement_vs_baseline': avg_return(gamma01) - avg_return(baseline_seq)
            },
            'gamma_0.5': {
                'avg_annual_return': avg_return(gamma05),
                'improvement_vs_baseline': avg_return(gamma05) - avg_return(baseline_seq)
            },
            'dqn': {
                'avg_annual_return': avg_return(dqn_results)
            },
            'td3': {
                'avg_annual_return': avg_return(td3_results)
            }
        }
    
    def save_results(self):
        """Save results to JSON."""
        
        json_path = os.path.join(self.output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
        
        print(f"\nResults saved to: {json_path}")


def main():
    """Run all experiments."""
    
    runner = ComprehensiveExperimentRunner()
    results = runner.run_all_experiments()
    
    # Print final summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    
    if 'summary' in results:
        summary = results['summary']
        
        print("\nBaseline Sequential (gamma=0.3, iter=100, seq=7):")
        print(f"  Average Annual Return: {summary['baseline_sequential']['avg_annual_return']*100:.2f}%")
        
        print("\nAblation Results:")
        print(f"  750 iterations: {summary['iterations_750']['avg_annual_return']*100:.2f}% (change: {summary['iterations_750']['improvement_vs_baseline']*100:+.2f}pp)")
        print(f"  14-day sequence: {summary['sequence_14days']['avg_annual_return']*100:.2f}% (change: {summary['sequence_14days']['improvement_vs_baseline']*100:+.2f}pp)")
        print(f"  Gamma 0.1: {summary['gamma_0.1']['avg_annual_return']*100:.2f}% (change: {summary['gamma_0.1']['improvement_vs_baseline']*100:+.2f}pp)")
        print(f"  Gamma 0.5: {summary['gamma_0.5']['avg_annual_return']*100:.2f}% (change: {summary['gamma_0.5']['improvement_vs_baseline']*100:+.2f}pp)")
        
        print("\nDQN/TD3 Comparison:")
        print(f"  DQN: {summary['dqn']['avg_annual_return']*100:.2f}%")
        print(f"  TD3: {summary['td3']['avg_annual_return']*100:.2f}%")
    
    return results


if __name__ == "__main__":
    start_time = time.time()
    results = main()
    elapsed = time.time() - start_time
    print(f"\nTotal experiment time: {elapsed/3600:.2f} hours")