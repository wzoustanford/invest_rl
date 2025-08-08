"""
Run Sequential Supervised Learning with 14-day sequences
Testing on 2021, 2022, 2023, 2024 data with full logging
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
import sys

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
        
        # Setup logging to file
        self.log_file = open(f'{self.output_dir}training_log.txt', 'w', buffering=1)
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        self.log_print(f"Loaded {len(self.all_files)} data files")
        self.log_print(f"Device: {self.device}")
        self.log_print(f"Output: {self.output_dir}")
        self.log_print("="*80)
        
        self.start_time = time.time()
    
    def log_print(self, msg):
        """Print to both stdout and log file with flush."""
        print(msg, flush=True)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        sys.stdout.flush()
        sys.stderr.flush()
    
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
        """Train with 14-day sequences and 100 iterations with detailed logging."""
        
        # Load up to 14 files
        self.log_print(f"    Loading training files for {year}-{month:02d}...")
        data_seq = []
        for f in training_files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    data['trainFeature'] = data['trainFeature'].to(self.device)
                    data['train_in_portfolio_series'] = data['train_in_portfolio_series'].to(self.device)
                    data_seq.append(data)
        
        if len(data_seq) < 7:  # Need at least 7 days
            self.log_print(f"    ERROR: Only {len(data_seq)} files, need at least 7")
            return None
        
        num_days = len(data_seq)
        self.log_print(f"    Loaded {num_days} days of data")
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with 100 iterations
        self.log_print(f"    Training with {num_days}-day sequence...")
        train_start = time.time()
        
        model.train()
        best_sharpe = -float('inf')
        
        for step in range(100):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            current_sharpe = 0
            
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
                
                if i == num_days - 1:  # Last file
                    current_sharpe = sharpe.item()
                
                # Gamma discounting over longer sequence
                gamma_power = 0.3 ** (num_days - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
            
            # Detailed progress every 20 iterations
            if (step + 1) % 20 == 0:
                elapsed = time.time() - train_start
                self.log_print(f"      Iter {step+1}/100 | Loss: {total_loss.item():.4f} | "
                             f"Sharpe: {current_sharpe:.3f} | Best: {best_sharpe:.3f} | "
                             f"Time: {elapsed:.1f}s")
        
        train_time = time.time() - train_start
        self.log_print(f"    Training complete in {train_time:.1f}s | Final Sharpe: {current_sharpe:.3f}")
        
        return model
    
    def evaluate_model(self, model, test_file, year, month):
        """Evaluate model with detailed logging."""
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
            
            # Calculate returns
            raw_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            net_return = raw_return - 0.0015
            
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = net_return / (torch.std(daily_returns) + 1e-10)
            volatility = torch.std(daily_returns)
            
            # Log evaluation details
            self.log_print(f"    Evaluation Results:")
            self.log_print(f"      Raw Return: {raw_return.item()*100:+.3f}%")
            self.log_print(f"      Transaction Cost: -0.15%")
            self.log_print(f"      Net Return: {net_return.item()*100:+.3f}%")
            self.log_print(f"      Sharpe Ratio: {sharpe.item():.3f}")
            self.log_print(f"      Daily Volatility: {volatility.item()*100:.3f}%")
            
            return {
                'return': net_return.item(),
                'sharpe': sharpe.item(),
                'raw_return': raw_return.item(),
                'volatility': volatility.item()
            }
    
    def run_year(self, year):
        """Run year with 14-day sequences and comprehensive logging."""
        
        self.log_print(f"\n{'='*70}")
        self.log_print(f"PROCESSING YEAR {year} - 14-DAY SEQUENCES")
        self.log_print(f"{'='*70}")
        
        year_start = time.time()
        
        results = {
            'year': year,
            'sequence_days': 14,
            'iterations': 100,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            self.log_print(f"\n  Month {month:02d}/{year}:")
            self.log_print(f"  " + "-"*50)
            
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                self.log_print(f"    Skipping: Insufficient data")
                continue
            
            # Get up to 14 training files (or all available if less)
            start_idx = max(0, file_idx - 13)
            training_files = self.all_files[start_idx:file_idx+1]
            test_date = self.get_date(test_file)
            self.log_print(f"    Test date: {test_date}")
            self.log_print(f"    File index: {file_idx}")
            self.log_print(f"    Using files {start_idx} to {file_idx} ({len(training_files)} files)")
            
            # Train
            model = self.train_model_14days(training_files, year, month)
            
            if model is None:
                self.log_print(f"    ERROR: Training failed")
                continue
            
            # Evaluate
            result = self.evaluate_model(model, test_file, year, month)
            
            if result:
                ret = result['return']
                results['cumulative'] *= (1 + ret)
                results['monthly_trades'].append({
                    'month': month,
                    'date': test_date,
                    'return': ret,
                    'sharpe': result['sharpe'],
                    'raw_return': result['raw_return'],
                    'volatility': result['volatility'],
                    'ytd': results['cumulative'] - 1
                })
                
                self.log_print(f"    YTD Performance: {(results['cumulative']-1)*100:+.2f}%")
            else:
                self.log_print(f"    ERROR: Evaluation failed")
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save progress
            with open(f'{self.output_dir}{year}_progress.json', 'w') as f:
                json.dump(results, f, indent=2, default=float)
        
        # Year statistics
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
            
            self.log_print(f"\n  {year} SUMMARY:")
            self.log_print(f"  " + "-"*50)
            self.log_print(f"    Annual Return: {results['annual_return']*100:+.2f}%")
            self.log_print(f"    Trades Executed: {results['trades']}/12")
            self.log_print(f"    Win Rate: {results['win_rate']*100:.1f}%")
            self.log_print(f"    Average Monthly: {results['avg_return']*100:+.2f}%")
            self.log_print(f"    Std Dev: {results['std_return']*100:.2f}%")
            self.log_print(f"    Best Month: {results['best_month']*100:+.2f}%")
            self.log_print(f"    Worst Month: {results['worst_month']*100:+.2f}%")
            self.log_print(f"    Sharpe Ratio: {results['sharpe']:.3f}")
            self.log_print(f"    Processing Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all(self):
        """Run all years with 14-day sequences."""
        
        self.log_print("\n" + "="*80)
        self.log_print("14-DAY SEQUENCE EXPERIMENT")
        self.log_print("Testing longer temporal context (14 days vs 7 days baseline)")
        self.log_print("Parameters: gamma=0.3, iterations=100, sequence=14 days")
        self.log_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print("="*80)
        
        all_results = {}
        
        # Run all years
        for year in [2021, 2022, 2023, 2024]:
            year_results = self.run_year(year)
            all_results[year] = year_results
            
            # Save individual year
            with open(f'{self.output_dir}{year}_results.json', 'w') as f:
                json.dump(year_results, f, indent=2, default=float)
        
        # Save all results
        with open(f'{self.output_dir}all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        # Print comparison
        self.print_comparison(all_results)
        
        # Close log file
        self.log_file.close()
        
        return all_results
    
    def print_comparison(self, results):
        """Print comparison with baselines."""
        
        self.log_print("\n" + "="*80)
        self.log_print("FINAL COMPARISON: 14-DAY SEQUENCES vs BASELINES")
        self.log_print("="*80)
        
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
        
        self.log_print(f"\n{'Year':<6} | {'7-day Base':>12} | {'750 iter':>12} | {'14-day Seq':>12} | {'vs Base':>10} | {'vs 750':>10}")
        self.log_print("-"*80)
        
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
                
                self.log_print(f"{year:<6} | {base_str} | {ret_750_str} | {new_str} | {diff_base_str} | {diff_750_str}")
        
        self.log_print("-"*80)
        
        # Averages
        if improvements_vs_base:
            avg_vs_base = np.mean(improvements_vs_base)
            self.log_print(f"Avg improvement vs 7-day baseline: {avg_vs_base:+.2f}pp")
        
        if improvements_vs_750:
            avg_vs_750 = np.mean(improvements_vs_750)
            self.log_print(f"Avg difference vs 750 iterations: {avg_vs_750:+.2f}pp")
        
        total_time = time.time() - self.start_time
        self.log_print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        self.log_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print(f"Results saved to: {self.output_dir}")
        
        # Print to stdout for visibility
        print(f"\n{'='*60}", flush=True)
        print("EXECUTION COMPLETE", flush=True)
        print(f"Total time: {total_time/60:.1f} minutes", flush=True)
        print(f"Log file: {self.output_dir}training_log.txt", flush=True)
        print(f"Results: {self.output_dir}all_results.json", flush=True)
        print(f"{'='*60}", flush=True)


def main():
    """Run 14-day sequence experiment with logging."""
    
    trainer = Sequential14DayTrainer()
    results = trainer.run_all()
    
    return results


if __name__ == "__main__":
    results = main()