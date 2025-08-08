"""
Complete 750 iteration experiment for 2023 and 2024 only
(2021 and 2022 already completed)
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime
import time
import gc

from model.iimodel import IIMODEL

class Complete750For2023_2024:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/750iter_2023_2024_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print("="*80)
        
        self.start_time = time.time()
    
    def get_date(self, filename):
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    def find_month_file(self, year, month):
        for i, filepath in enumerate(self.all_files):
            date_str = self.get_date(filepath)
            if date_str:
                file_year = int(date_str[:4])
                file_month = int(date_str[5:7])
                if file_year == year and file_month == month:
                    return i, filepath
        return None, None
    
    def train_model_750(self, training_files, year, month):
        """Train with 750 iterations showing progress."""
        
        # Load data
        data_seq = []
        for f in training_files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    data['trainFeature'] = data['trainFeature'].to(self.device)
                    data['train_in_portfolio_series'] = data['train_in_portfolio_series'].to(self.device)
                    data_seq.append(data)
        
        if len(data_seq) != 7:
            return None
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training
        print(f"    Training {year}-{month:02d}: ", end='', flush=True)
        
        model.train()
        for step in range(750):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            
            for i in range(7):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                gamma_power = 0.3 ** (7 - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            # Progress indicator
            if (step + 1) % 150 == 0:
                print(f"{step+1}", end=' ', flush=True)
        
        print("Done!")
        return model
    
    def evaluate_model(self, model, test_file):
        """Evaluate model."""
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
            ret = torch.sum((series[:, -1:] - series[:, 0:1]) * shares) - 0.0015
            
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = ret / (torch.std(daily_returns) + 1e-10)
            
            return {'return': ret.item(), 'sharpe': sharpe.item()}
    
    def run_year(self, year):
        """Run year with 750 iterations."""
        
        print(f"\n{'='*70}")
        print(f"YEAR {year} - 750 ITERATIONS")
        print(f"{'='*70}")
        
        year_start = time.time()
        
        results = {
            'year': year,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                print(f"  Month {month:02d}: No data")
                continue
            
            training_files = self.all_files[file_idx-6:file_idx+1]
            test_date = self.get_date(test_file)
            
            # Train
            model = self.train_model_750(training_files, year, month)
            
            if model is None:
                print(f"  Month {month:02d}: Training failed")
                continue
            
            # Evaluate
            result = self.evaluate_model(model, test_file)
            
            if result:
                ret = result['return']
                results['cumulative'] *= (1 + ret)
                results['monthly_trades'].append({
                    'month': month,
                    'date': test_date,
                    'return': ret,
                    'sharpe': result['sharpe'],
                    'ytd': results['cumulative'] - 1
                })
                
                print(f"    Return: {ret*100:+.2f}% | YTD: {(results['cumulative']-1)*100:+.2f}%")
            else:
                print(f"  Month {month:02d}: Eval failed")
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Statistics
        if results['monthly_trades']:
            returns = [t['return'] for t in results['monthly_trades']]
            results['annual_return'] = results['cumulative'] - 1
            results['avg_return'] = np.mean(returns)
            results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            results['trades'] = len(returns)
            
            year_time = time.time() - year_start
            print(f"\n  {year} Summary:")
            print(f"    Annual Return: {results['annual_return']*100:+.2f}%")
            print(f"    Trades: {results['trades']}/12")
            print(f"    Win Rate: {results['win_rate']*100:.0f}%")
            print(f"    Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all(self):
        """Run 2023 and 2024."""
        
        print("\n" + "="*70)
        print("COMPLETING 750 ITERATION EXPERIMENT")
        print("Running: 2023 and 2024 (2021-2022 already complete)")
        print("="*70)
        
        all_results = {
            # Include previous results
            '2021': {
                'annual_return': -0.1622,
                'trades': 12,
                'win_rate': 0.333,
                'source': 'previous_run'
            },
            '2022': {
                'annual_return': 0.1400,
                'trades': 12,
                'win_rate': 0.417,
                'source': 'previous_run'
            }
        }
        
        # Run 2023 and 2024
        for year in [2023, 2024]:
            year_results = self.run_year(year)
            all_results[year] = year_results
            
            # Save
            with open(f'{self.output_dir}{year}_results.json', 'w') as f:
                json.dump(year_results, f, indent=2, default=float)
        
        # Print final comparison
        self.print_final_comparison(all_results)
        
        # Save all results
        with open(f'{self.output_dir}final_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        return all_results
    
    def print_final_comparison(self, results):
        """Print comparison with baseline."""
        
        print("\n" + "="*70)
        print("FINAL COMPARISON: 750 ITERATIONS vs BASELINE (100 iter)")
        print("="*70)
        
        baseline = {
            2021: -0.3379,
            2022: 0.1387,
            2023: 0.1088,
            2024: None  # No baseline yet
        }
        
        print(f"\n{'Year':<6} | {'Baseline':>15} | {'750 iter':>15} | {'Improvement':>12}")
        print("-"*70)
        
        for year in [2021, 2022, 2023, 2024]:
            if year in results:
                if 'annual_return' in results[year]:
                    new_ret = results[year]['annual_return']
                else:
                    new_ret = results[str(year)].get('annual_return', 0)
                
                if year in baseline and baseline[year] is not None:
                    base_ret = baseline[year]
                    diff = (new_ret - base_ret) * 100
                    print(f"{year:<6} | {base_ret*100:>14.2f}% | {new_ret*100:>14.2f}% | {diff:+11.2f}pp")
                else:
                    print(f"{year:<6} | {'N/A':>15} | {new_ret*100:>14.2f}% | {'New':>12}")
        
        print("-"*70)
        
        # Average improvement for 2021-2023
        improvements = []
        for year in [2021, 2022, 2023]:
            if year in baseline and baseline[year] is not None:
                if year in results:
                    if 'annual_return' in results[year]:
                        new = results[year]['annual_return']
                    else:
                        new = results[str(year)].get('annual_return', 0)
                    improvements.append((new - baseline[year]) * 100)
        
        if improvements:
            print(f"Avg Improvement (2021-2023): {np.mean(improvements):+.2f} percentage points")
        
        total_time = time.time() - self.start_time
        print(f"\nTotal time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")


def main():
    runner = Complete750For2023_2024()
    return runner.run_all()


if __name__ == "__main__":
    results = main()