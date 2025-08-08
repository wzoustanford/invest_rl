"""
Run the specific experiments requested:
1. 2021 baseline for all algorithms
2. Sequential with 750 iterations
3. Sequential with 14-day sequences
4. Sequential with gamma 0.1 and 0.5
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime
from typing import Dict, List

from model.iimodel import IIMODEL

class RequestedExperiments:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/requested_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data files
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files")
        
        # Results storage
        self.results = {}
    
    def find_month_file(self, year: int, month: int):
        """Find file index for specific year and month."""
        for i, filepath in enumerate(self.all_files):
            if f'test_data_start_date_{year}_{month:02d}' in filepath or \
               f'test_data_start_date_{year}-{month:02d}' in filepath:
                return i, filepath
        return None, None
    
    def train_sequential(self, training_files: List[str], gamma: float, iterations: int, sequence_days: int):
        """Train Sequential Supervised model."""
        
        # Load data files
        data_seq = []
        for i in range(min(sequence_days, len(training_files))):
            if os.path.exists(training_files[i]):
                with open(training_files[i], 'rb') as f:
                    data_seq.append(pickle.load(f))
        
        if len(data_seq) < min(7, sequence_days):
            return None
        
        # Initialize model
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop - FULL iterations as requested
        for step in range(iterations):
            model.train()
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0).to(self.device)
            
            for i in range(len(data_seq)):
                features = data_seq[i]['trainFeature'].to(self.device)
                series = data_seq[i]['train_in_portfolio_series'].to(self.device)
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                # Gamma discounting
                gamma_power = gamma ** (len(data_seq) - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            # Progress indicator
            if (step + 1) % 100 == 0:
                print(f"    Step {step+1}/{iterations}", end='\r')
        
        return model
    
    def evaluate_model(self, model, test_file):
        """Evaluate model on test data."""
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
            
            return {'return': ret.item(), 'sharpe': sharpe.item()}
    
    def run_year_monthly(self, year: int, gamma: float, iterations: int, sequence_days: int):
        """Run 12 monthly trades for a year."""
        
        print(f"\n  Year {year} (gamma={gamma}, iter={iterations}, seq={sequence_days}):")
        
        year_results = {
            'trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < sequence_days - 1:
                continue
            
            # Get training files
            start_idx = max(0, file_idx - sequence_days + 1)
            training_files = self.all_files[start_idx:file_idx + 1]
            
            # Train model
            model = self.train_sequential(training_files, gamma, iterations, sequence_days)
            
            # Evaluate
            result = self.evaluate_model(model, test_file)
            
            if result:
                year_results['cumulative'] *= (1 + result['return'])
                year_results['trades'].append({
                    'month': month,
                    'return': result['return'],
                    'sharpe': result['sharpe']
                })
                
                print(f"    Month {month:02d}: {result['return']*100:+.2f}%", end='  ')
                if month % 4 == 0:
                    print()
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
        
        # Calculate statistics
        if year_results['trades']:
            returns = [t['return'] for t in year_results['trades']]
            year_results['annual_return'] = year_results['cumulative'] - 1
            year_results['num_trades'] = len(returns)
            year_results['avg_return'] = np.mean(returns)
            year_results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            
            print(f"\n    Annual: {year_results['annual_return']*100:+.2f}%, Trades: {year_results['num_trades']}, Win Rate: {year_results['win_rate']*100:.0f}%")
        
        return year_results
    
    def run_all_experiments(self):
        """Run all requested experiments."""
        
        print("\n" + "="*80)
        print("RUNNING REQUESTED EXPERIMENTS")
        print("="*80)
        
        # 1. BASELINE: 2021 with standard parameters
        print("\n1. BASELINE - 2021 Performance")
        print("-"*40)
        self.results['2021_baseline'] = self.run_year_monthly(2021, gamma=0.3, iterations=100, sequence_days=7)
        
        # 2. TEST 750 ITERATIONS
        print("\n\n2. ABLATION - 750 Iterations")
        print("-"*40)
        self.results['iter_750'] = {}
        for year in [2021, 2022, 2023]:
            self.results['iter_750'][year] = self.run_year_monthly(year, gamma=0.3, iterations=750, sequence_days=7)
        
        # 3. TEST 14-DAY SEQUENCES
        print("\n\n3. ABLATION - 14-Day Sequences")
        print("-"*40)
        self.results['seq_14'] = {}
        for year in [2021, 2022, 2023]:
            self.results['seq_14'][year] = self.run_year_monthly(year, gamma=0.3, iterations=100, sequence_days=14)
        
        # 4. TEST GAMMA 0.1
        print("\n\n4. ABLATION - Gamma 0.1")
        print("-"*40)
        self.results['gamma_0.1'] = {}
        for year in [2021, 2022, 2023]:
            self.results['gamma_0.1'][year] = self.run_year_monthly(year, gamma=0.1, iterations=100, sequence_days=7)
        
        # 5. TEST GAMMA 0.5
        print("\n\n5. ABLATION - Gamma 0.5")
        print("-"*40)
        self.results['gamma_0.5'] = {}
        for year in [2021, 2022, 2023]:
            self.results['gamma_0.5'][year] = self.run_year_monthly(year, gamma=0.5, iterations=100, sequence_days=7)
        
        # Save results
        with open(f'{self.output_dir}results.json', 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return self.results
    
    def print_summary(self):
        """Print summary comparison."""
        
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        # Extract baseline performance (2022-2023 from previous experiments)
        baseline_2022 = 0.1387  # From previous experiments
        baseline_2023 = 0.1088  # From previous experiments
        
        print("\n" + "-"*80)
        print("Configuration         | 2021     | 2022     | 2023     | Average  | Notes")
        print("-"*80)
        
        # Baseline (from 2021 + previous 2022/2023)
        baseline_2021 = self.results.get('2021_baseline', {}).get('annual_return', 0)
        baseline_avg = (baseline_2021 + baseline_2022 + baseline_2023) / 3
        print(f"Baseline (γ=0.3)      | {baseline_2021*100:+7.2f}% | {baseline_2022*100:+7.2f}% | {baseline_2023*100:+7.2f}% | {baseline_avg*100:+7.2f}% | 100 iter, 7 days")
        
        # 750 iterations
        if 'iter_750' in self.results:
            returns = [self.results['iter_750'].get(y, {}).get('annual_return', 0) for y in [2021, 2022, 2023]]
            avg = np.mean(returns)
            print(f"750 Iterations        | {returns[0]*100:+7.2f}% | {returns[1]*100:+7.2f}% | {returns[2]*100:+7.2f}% | {avg*100:+7.2f}% | 750 iter, 7 days")
        
        # 14-day sequences
        if 'seq_14' in self.results:
            returns = [self.results['seq_14'].get(y, {}).get('annual_return', 0) for y in [2021, 2022, 2023]]
            avg = np.mean(returns)
            print(f"14-Day Sequences      | {returns[0]*100:+7.2f}% | {returns[1]*100:+7.2f}% | {returns[2]*100:+7.2f}% | {avg*100:+7.2f}% | 100 iter, 14 days")
        
        # Gamma 0.1
        if 'gamma_0.1' in self.results:
            returns = [self.results['gamma_0.1'].get(y, {}).get('annual_return', 0) for y in [2021, 2022, 2023]]
            avg = np.mean(returns)
            print(f"Gamma 0.1             | {returns[0]*100:+7.2f}% | {returns[1]*100:+7.2f}% | {returns[2]*100:+7.2f}% | {avg*100:+7.2f}% | γ=0.1, 100 iter")
        
        # Gamma 0.5
        if 'gamma_0.5' in self.results:
            returns = [self.results['gamma_0.5'].get(y, {}).get('annual_return', 0) for y in [2021, 2022, 2023]]
            avg = np.mean(returns)
            print(f"Gamma 0.5             | {returns[0]*100:+7.2f}% | {returns[1]*100:+7.2f}% | {returns[2]*100:+7.2f}% | {avg*100:+7.2f}% | γ=0.5, 100 iter")
        
        print("-"*80)


def main():
    """Run all requested experiments."""
    
    runner = RequestedExperiments()
    results = runner.run_all_experiments()
    runner.print_summary()
    
    return results


if __name__ == "__main__":
    import time
    start = time.time()
    results = main()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/60:.1f} minutes")