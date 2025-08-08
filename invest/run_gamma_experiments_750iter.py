"""
Test different gamma values (0.1 and 0.5) with 750 iterations
7-day sequences as requested
Final model improvement experiment
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

class GammaExperiment750:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/gamma_750iter_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        self.log_file = open(f'{self.output_dir}training_log.txt', 'w', buffering=1)
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        self.log_print(f"Loaded {len(self.all_files)} data files")
        self.log_print(f"Device: {self.device}")
        self.log_print(f"Output: {self.output_dir}")
        self.log_print("="*80)
        
        self.start_time = time.time()
    
    def log_print(self, msg, end='\n'):
        """Print to both stdout and log file."""
        print(msg, end=end, flush=True)
        self.log_file.write(msg + (end if end else ''))
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
    
    def train_model_gamma(self, training_files, gamma, year, month):
        """Train with specific gamma value and 750 iterations."""
        
        # Load 7 files
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
        
        # Training with 750 iterations
        self.log_print(f"      Training {year}-{month:02d} (gamma={gamma}): ", end='')
        train_start = time.time()
        
        model.train()
        for step in range(750):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Process 7 files with specified gamma
            for i in range(7):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                # Apply gamma discounting
                gamma_power = gamma ** (7 - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            # Progress every 150 iterations
            if (step + 1) % 150 == 0:
                self.log_print(f"{step+1} ", end='')
        
        train_time = time.time() - train_start
        self.log_print(f"Done! [{train_time:.1f}s]")
        
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
            
            raw_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            net_return = raw_return - 0.0015
            
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = net_return / (torch.std(daily_returns) + 1e-10)
            
            return {
                'return': net_return.item(),
                'sharpe': sharpe.item(),
                'raw_return': raw_return.item()
            }
    
    def run_year_gamma(self, year, gamma):
        """Run year with specific gamma value."""
        
        self.log_print(f"\n  Processing Year {year} with Gamma={gamma}")
        self.log_print(f"  " + "-"*60)
        
        year_start = time.time()
        
        results = {
            'year': year,
            'gamma': gamma,
            'iterations': 750,
            'sequence_days': 7,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                self.log_print(f"    Month {month:02d}: No data")
                continue
            
            # Get 7 training files
            training_files = self.all_files[file_idx-6:file_idx+1]
            test_date = self.get_date(test_file)
            
            # Train
            model = self.train_model_gamma(training_files, gamma, year, month)
            
            if model is None:
                self.log_print(f"    Month {month:02d}: Training failed")
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
                
                self.log_print(f"      Return: {ret*100:+.2f}% | YTD: {(results['cumulative']-1)*100:+.2f}%")
            else:
                self.log_print(f"    Month {month:02d}: Eval failed")
            
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
            
            year_time = time.time() - year_start
            self.log_print(f"\n    {year} Summary (Gamma={gamma}):")
            self.log_print(f"      Annual Return: {results['annual_return']*100:+.2f}%")
            self.log_print(f"      Win Rate: {results['win_rate']*100:.0f}%")
            self.log_print(f"      Trades: {results['trades']}/12")
            self.log_print(f"      Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all_experiments(self):
        """Run experiments for both gamma values."""
        
        self.log_print("\n" + "="*80)
        self.log_print("GAMMA EXPERIMENTS - 750 ITERATIONS, 7-DAY SEQUENCES")
        self.log_print("Testing Gamma values: 0.1 (less discounting) and 0.5 (more discounting)")
        self.log_print("Baseline Gamma: 0.3")
        self.log_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print("="*80)
        
        all_results = {}
        
        # Test both gamma values
        for gamma in [0.1, 0.5]:
            self.log_print(f"\n{'='*70}")
            self.log_print(f"TESTING GAMMA = {gamma}")
            self.log_print(f"{'='*70}")
            
            gamma_results = {}
            
            # Run all years
            for year in [2021, 2022, 2023, 2024]:
                year_results = self.run_year_gamma(year, gamma)
                gamma_results[year] = year_results
                
                # Save individual results
                with open(f'{self.output_dir}gamma{gamma}_{year}_results.json', 'w') as f:
                    json.dump(year_results, f, indent=2, default=float)
            
            all_results[f'gamma_{gamma}'] = gamma_results
        
        # Save all results
        with open(f'{self.output_dir}all_gamma_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        # Print comprehensive comparison
        self.print_final_comparison(all_results)
        
        # Close log
        self.log_file.close()
        
        return all_results
    
    def print_final_comparison(self, results):
        """Print comparison of all gamma values."""
        
        self.log_print("\n" + "="*80)
        self.log_print("FINAL COMPARISON: GAMMA EXPERIMENTS")
        self.log_print("="*80)
        
        # Previous results for comparison
        baseline_gamma03 = {
            2021: -0.1622,  # 750 iter, gamma=0.3
            2022: 0.1400,
            2023: 4.0617,
            2024: -0.3047
        }
        
        self.log_print(f"\n{'Year':<6} | {'Gamma=0.3 (base)':>16} | {'Gamma=0.1':>16} | {'Gamma=0.5':>16} | {'Best':>10}")
        self.log_print("-"*80)
        
        for year in [2021, 2022, 2023, 2024]:
            base = baseline_gamma03.get(year, 0)
            
            # Get results for each gamma
            g01 = results.get('gamma_0.1', {}).get(year, {}).get('annual_return', 0)
            g05 = results.get('gamma_0.5', {}).get(year, {}).get('annual_return', 0)
            
            # Find best
            returns = {'0.3': base, '0.1': g01, '0.5': g05}
            best_gamma = max(returns, key=returns.get)
            
            self.log_print(f"{year:<6} | {base*100:>15.2f}% | {g01*100:>15.2f}% | {g05*100:>15.2f}% | γ={best_gamma:>8}")
        
        self.log_print("-"*80)
        
        # Calculate averages
        avg_g03 = np.mean([baseline_gamma03[y] for y in [2021, 2022, 2023] if y in baseline_gamma03])
        avg_g01 = np.mean([results.get('gamma_0.1', {}).get(y, {}).get('annual_return', 0) 
                          for y in [2021, 2022, 2023]])
        avg_g05 = np.mean([results.get('gamma_0.5', {}).get(y, {}).get('annual_return', 0) 
                          for y in [2021, 2022, 2023]])
        
        self.log_print(f"{'3-Yr Avg':<6} | {avg_g03*100:>15.2f}% | {avg_g01*100:>15.2f}% | {avg_g05*100:>15.2f}% |")
        self.log_print("-"*80)
        
        # Analysis
        self.log_print("\nKey Insights:")
        self.log_print(f"  Gamma=0.1 (less discounting): Focus on recent data")
        self.log_print(f"  Gamma=0.3 (baseline): Balanced temporal weighting")
        self.log_print(f"  Gamma=0.5 (more discounting): Strong emphasis on most recent day")
        
        # Determine overall winner
        if avg_g01 > avg_g03 and avg_g01 > avg_g05:
            self.log_print(f"\n  WINNER: Gamma=0.1 with {avg_g01*100:.2f}% average return")
        elif avg_g05 > avg_g03 and avg_g05 > avg_g01:
            self.log_print(f"\n  WINNER: Gamma=0.5 with {avg_g05*100:.2f}% average return")
        else:
            self.log_print(f"\n  WINNER: Gamma=0.3 (baseline) with {avg_g03*100:.2f}% average return")
        
        total_time = time.time() - self.start_time
        self.log_print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        self.log_print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print(f"Results saved to: {self.output_dir}")
        
        # Print summary to stdout
        print("\n" + "="*60)
        print("GAMMA EXPERIMENT COMPLETE")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Results: {self.output_dir}")
        print("="*60)


def main():
    """Run gamma experiments."""
    
    experiment = GammaExperiment750()
    results = experiment.run_all_experiments()
    
    return results


if __name__ == "__main__":
    results = main()