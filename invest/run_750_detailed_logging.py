"""
750 iteration experiment with detailed logging and progress tracking
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

class DetailedTracker750:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/750iter_detailed_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = open(f'{self.output_dir}training_log.txt', 'w')
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        self.log_print(f"Loaded {len(self.all_files)} data files")
        self.log_print(f"Device: {self.device}")
        self.log_print(f"Output: {self.output_dir}")
        self.log_print("="*80)
        
        self.start_time = time.time()
        self.trades_completed = 0
        self.total_trades = 48  # 4 years * 12 months
    
    def log_print(self, msg):
        """Print to both stdout and log file."""
        print(msg, flush=True)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
    
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
    
    def train_with_detailed_progress(self, training_files, year, month):
        """Train with detailed iteration progress."""
        
        # Load data
        self.log_print(f"  Loading 7 training files...")
        data_seq = []
        for f in training_files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    data['trainFeature'] = data['trainFeature'].to(self.device)
                    data['train_in_portfolio_series'] = data['train_in_portfolio_series'].to(self.device)
                    data_seq.append(data)
        
        if len(data_seq) != 7:
            self.log_print(f"  ERROR: Only loaded {len(data_seq)} files, need 7")
            return None
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with detailed progress
        self.log_print(f"  Starting training for {year}-{month:02d} (750 iterations)")
        train_start = time.time()
        
        model.train()
        best_sharpe = -float('inf')
        
        for step in range(750):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            current_sharpe = 0
            
            for i in range(7):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                if i == 6:  # Last file
                    current_sharpe = sharpe.item()
                
                gamma_power = 0.3 ** (7 - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            # Update best sharpe
            if current_sharpe > best_sharpe:
                best_sharpe = current_sharpe
            
            # Detailed progress every 50 iterations
            if (step + 1) % 50 == 0:
                elapsed = time.time() - train_start
                rate = (step + 1) / elapsed
                eta = (750 - step - 1) / rate
                self.log_print(f"    Iter {step+1:3d}/750 | Loss: {total_loss.item():+.4f} | "
                             f"Sharpe: {current_sharpe:.3f} | Best: {best_sharpe:.3f} | "
                             f"Time: {elapsed:.1f}s | ETA: {eta:.1f}s")
        
        train_time = time.time() - train_start
        self.log_print(f"  Training complete in {train_time:.1f}s | Final Sharpe: {current_sharpe:.3f}")
        
        return model
    
    def evaluate_model(self, model, test_file, year, month):
        """Evaluate with detailed output."""
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
            
            # Calculate raw return
            raw_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            
            # Apply transaction cost
            net_return = raw_return - 0.0015
            
            # Calculate Sharpe
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = net_return / (torch.std(daily_returns) + 1e-10)
            
            # Log detailed evaluation results
            self.log_print(f"  Evaluation for {year}-{month:02d}:")
            self.log_print(f"    Raw Return: {raw_return.item()*100:+.3f}%")
            self.log_print(f"    Transaction Cost: -0.15%")
            self.log_print(f"    Net Return: {net_return.item()*100:+.3f}%")
            self.log_print(f"    Sharpe Ratio: {sharpe.item():.3f}")
            self.log_print(f"    Daily Volatility: {torch.std(daily_returns).item()*100:.3f}%")
            
            return {
                'return': net_return.item(),
                'sharpe': sharpe.item(),
                'raw_return': raw_return.item(),
                'volatility': torch.std(daily_returns).item()
            }
    
    def run_year(self, year):
        """Run year with comprehensive logging."""
        
        self.log_print(f"\n{'='*80}")
        self.log_print(f"PROCESSING YEAR {year}")
        self.log_print(f"{'='*80}")
        
        year_start = time.time()
        
        results = {
            'year': year,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            self.trades_completed += 1
            elapsed = time.time() - self.start_time
            
            self.log_print(f"\n{'='*60}")
            self.log_print(f"Month {month:02d}/{year} | Trade {self.trades_completed}/{self.total_trades}")
            self.log_print(f"Total Elapsed: {elapsed/60:.1f} min")
            
            if self.trades_completed > 1:
                avg_time_per_trade = elapsed / (self.trades_completed - 1)
                remaining_trades = self.total_trades - self.trades_completed + 1
                eta = avg_time_per_trade * remaining_trades
                self.log_print(f"ETA: {eta/60:.1f} min remaining")
            
            self.log_print(f"{'='*60}")
            
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                self.log_print(f"  Skipping {year}-{month:02d}: Insufficient data")
                continue
            
            training_files = self.all_files[file_idx-6:file_idx+1]
            test_date = self.get_date(test_file)
            self.log_print(f"  Test date: {test_date}")
            self.log_print(f"  File index: {file_idx}")
            
            # Train
            model = self.train_with_detailed_progress(training_files, year, month)
            
            if model is None:
                self.log_print(f"  ERROR: Training failed for {year}-{month:02d}")
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
                
                self.log_print(f"  YTD Performance: {(results['cumulative']-1)*100:+.2f}%")
            else:
                self.log_print(f"  ERROR: Evaluation failed for {year}-{month:02d}")
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
            # Save intermediate results
            with open(f'{self.output_dir}{year}_progress.json', 'w') as f:
                json.dump(results, f, indent=2, default=float)
        
        # Year summary
        if results['monthly_trades']:
            returns = [t['return'] for t in results['monthly_trades']]
            results['annual_return'] = results['cumulative'] - 1
            results['avg_return'] = np.mean(returns)
            results['std_return'] = np.std(returns)
            results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            results['trades'] = len(returns)
            results['best_month'] = max(returns)
            results['worst_month'] = min(returns)
            
            year_time = time.time() - year_start
            
            self.log_print(f"\n{'='*60}")
            self.log_print(f"{year} SUMMARY")
            self.log_print(f"{'='*60}")
            self.log_print(f"  Annual Return: {results['annual_return']*100:+.2f}%")
            self.log_print(f"  Trades Executed: {results['trades']}/12")
            self.log_print(f"  Win Rate: {results['win_rate']*100:.1f}%")
            self.log_print(f"  Average Monthly: {results['avg_return']*100:+.2f}%")
            self.log_print(f"  Std Dev: {results['std_return']*100:.2f}%")
            self.log_print(f"  Best Month: {results['best_month']*100:+.2f}%")
            self.log_print(f"  Worst Month: {results['worst_month']*100:+.2f}%")
            self.log_print(f"  Processing Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all(self):
        """Run all years with comprehensive logging."""
        
        self.log_print("\n" + "="*80)
        self.log_print("750 ITERATION EXPERIMENT - DETAILED LOGGING")
        self.log_print("="*80)
        self.log_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print("Parameters: gamma=0.3, iterations=750, sequence=7 days")
        self.log_print("="*80)
        
        all_results = {}
        
        for year in [2021, 2022, 2023, 2024]:
            year_results = self.run_year(year)
            all_results[year] = year_results
            
            # Save cumulative results
            with open(f'{self.output_dir}results_cumulative.json', 'w') as f:
                json.dump(all_results, f, indent=2, default=float)
        
        # Final comparison
        self.print_final_comparison(all_results)
        
        # Close log file
        self.log_file.close()
        
        return all_results
    
    def print_final_comparison(self, results):
        """Print final comparison with baseline."""
        
        self.log_print("\n" + "="*80)
        self.log_print("FINAL RESULTS - 750 ITERATIONS vs BASELINE")
        self.log_print("="*80)
        
        baseline = {2021: -0.3379, 2022: 0.1387, 2023: 0.1088}
        
        self.log_print(f"\n{'Year':<6} | {'Baseline (100 iter)':>20} | {'750 iter':>20} | {'Improvement':>15}")
        self.log_print("-"*80)
        
        improvements = []
        for year in [2021, 2022, 2023, 2024]:
            if year in results and 'annual_return' in results[year]:
                new_ret = results[year]['annual_return']
                if year in baseline:
                    base_ret = baseline[year]
                    diff = (new_ret - base_ret) * 100
                    improvements.append(diff)
                    self.log_print(f"{year:<6} | {base_ret*100:>19.2f}% | {new_ret*100:>19.2f}% | {diff:+14.2f}pp")
                else:
                    self.log_print(f"{year:<6} | {'N/A':>20} | {new_ret*100:>19.2f}% | {'New data':>15}")
        
        if improvements:
            self.log_print("-"*80)
            avg_improvement = np.mean(improvements)
            self.log_print(f"Average improvement (2021-2023): {avg_improvement:+.2f} percentage points")
        
        total_time = time.time() - self.start_time
        self.log_print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        self.log_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print(f"Results saved to: {self.output_dir}")
        
        # Print to stdout as well for visibility
        print("\n" + "="*60)
        print("EXECUTION COMPLETE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Log file: {self.output_dir}training_log.txt")
        print(f"Results: {self.output_dir}results_cumulative.json")
        print("="*60)


def main():
    tracker = DetailedTracker750()
    return tracker.run_all()


if __name__ == "__main__":
    results = main()