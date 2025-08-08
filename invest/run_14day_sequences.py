"""
Run Sequential Supervised Learning with 14-day sequences
Testing on 2021, 2022, 2023, 2024 data
Step 2 of model improvements
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

class Sequential14DayTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/14day_seq_{timestamp}/'
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
    
    def train_model_14days(self, training_files, year, month):
        """Train with 14-day sequences and 100 iterations."""
        
        # Load up to 14 files
        data_seq = []
        for f in training_files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    data['trainFeature'] = data['trainFeature'].to(self.device)
                    data['train_in_portfolio_series'] = data['train_in_portfolio_series'].to(self.device)
                    data_seq.append(data)
        
        if len(data_seq) < 7:  # Need at least 7 days
            return None
        
        num_days = len(data_seq)
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with 100 iterations (standard)
        print(f"    Training {year}-{month:02d} ({num_days} days): ", end='', flush=True)
        
        model.train()
        for step in range(100):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Process all available files (up to 14) with gamma discounting
            for i in range(num_days):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                # Gamma discounting over longer sequence
                gamma_power = 0.3 ** (num_days - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            # Progress indicator
            if (step + 1) % 20 == 0:
                print(f"{step+1}", end=' ', flush=True)
        
        print(f"Done! (used {num_days} days)")
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
        """Run year with 14-day sequences."""
        
        print(f"\n{'='*70}")
        print(f"YEAR {year} - 14-DAY SEQUENCES")
        print(f"{'='*70}")
        
        year_start = time.time()
        
        results = {
            'year': year,
            'sequence_days': 14,
            'iterations': 100,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                print(f"  Month {month:02d}: No data")
                continue
            
            # Get up to 14 training files (or all available if less)
            start_idx = max(0, file_idx - 13)
            training_files = self.all_files[start_idx:file_idx+1]
            test_date = self.get_date(test_file)
            
            # Train
            model = self.train_model_14days(training_files, year, month)
            
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
            results['std_return'] = np.std(returns)
            results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            results['trades'] = len(returns)
            results['best_month'] = max(returns)
            results['worst_month'] = min(returns)
            results['sharpe'] = results['avg_return'] / (results['std_return'] + 1e-10)
            
            year_time = time.time() - year_start
            print(f"\n  {year} Summary:")
            print(f"    Annual Return: {results['annual_return']*100:+.2f}%")
            print(f"    Trades: {results['trades']}/12")
            print(f"    Win Rate: {results['win_rate']*100:.0f}%")
            print(f"    Avg Monthly: {results['avg_return']*100:+.2f}%")
            print(f"    Best Month: {results['best_month']*100:+.2f}%")
            print(f"    Worst Month: {results['worst_month']*100:+.2f}%")
            print(f"    Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all(self):
        """Run all years with 14-day sequences."""
        
        print("\n" + "="*70)
        print("14-DAY SEQUENCE EXPERIMENT")
        print("Testing longer temporal context (14 days vs 7 days baseline)")
        print("Parameters: gamma=0.3, iterations=100, sequence=14 days")
        print("="*70)
        
        all_results = {}
        
        # Run all years
        for year in [2021, 2022, 2023, 2024]:
            year_results = self.run_year(year)
            all_results[year] = year_results
            
            # Save individual year
            with open(f'{self.output_dir}{year}_results.json', 'w') as f:
                json.dump(year_results, f, indent=2, default=float)
        
        # Print comparison
        self.print_comparison(all_results)
        
        # Save all results
        with open(f'{self.output_dir}all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        return all_results
    
    def print_comparison(self, results):
        """Print comparison with baseline and 750 iterations."""
        
        print("\n" + "="*70)
        print("COMPARISON: 14-DAY SEQUENCES vs BASELINES")
        print("="*70)
        
        # Baseline results
        baseline_7day = {
            2021: -0.3379,
            2022: 0.1387,
            2023: 0.1088,
            2024: None
        }
        
        results_750iter = {
            2021: -0.1622,
            2022: 0.1400,
            2023: 4.0617,  # 406.17%
            2024: -0.3047
        }
        
        print(f"\n{'Year':<6} | {'7-day Base':>12} | {'750 iter':>12} | {'14-day Seq':>12} | {'vs Base':>10} | {'vs 750':>10}")
        print("-"*80)
        
        improvements_vs_base = []
        improvements_vs_750 = []
        
        for year in [2021, 2022, 2023, 2024]:
            if year in results and 'annual_return' in results[year]:
                new_ret = results[year]['annual_return']
                
                # Format strings
                if year in baseline_7day and baseline_7day[year] is not None:
                    base_ret = baseline_7day[year]
                    diff_base = (new_ret - base_ret) * 100
                    improvements_vs_base.append(diff_base)
                    base_str = f"{base_ret*100:>11.2f}%"
                    diff_base_str = f"{diff_base:+9.2f}pp"
                else:
                    base_str = "N/A"
                    diff_base_str = "N/A"
                
                if year in results_750iter:
                    ret_750 = results_750iter[year]
                    diff_750 = (new_ret - ret_750) * 100
                    improvements_vs_750.append(diff_750)
                    ret_750_str = f"{ret_750*100:>11.2f}%"
                    diff_750_str = f"{diff_750:+9.2f}pp"
                else:
                    ret_750_str = "N/A"
                    diff_750_str = "N/A"
                
                new_str = f"{new_ret*100:>11.2f}%"
                
                print(f"{year:<6} | {base_str} | {ret_750_str} | {new_str} | {diff_base_str} | {diff_750_str}")
        
        print("-"*80)
        
        # Averages
        if improvements_vs_base:
            avg_vs_base = np.mean(improvements_vs_base)
            print(f"Avg improvement vs 7-day baseline: {avg_vs_base:+.2f}pp")
        
        if improvements_vs_750:
            avg_vs_750 = np.mean(improvements_vs_750)
            print(f"Avg difference vs 750 iterations: {avg_vs_750:+.2f}pp")
        
        # Summary statistics for 14-day results
        print("\n14-Day Sequence Statistics (2021-2023):")
        years_with_data = [y for y in [2021, 2022, 2023] if y in results and 'annual_return' in results[y]]
        if years_with_data:
            returns = [results[y]['annual_return'] for y in years_with_data]
            print(f"  Mean Annual Return: {np.mean(returns)*100:+.2f}%")
            print(f"  Median Annual Return: {np.median(returns)*100:+.2f}%")
            print(f"  Best Year: {max(returns)*100:+.2f}%")
            print(f"  Worst Year: {min(returns)*100:+.2f}%")
        
        total_time = time.time() - self.start_time
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")


def main():
    """Run 14-day sequence experiment."""
    
    trainer = Sequential14DayTrainer()
    results = trainer.run_all()
    
    return results


if __name__ == "__main__":
    results = main()