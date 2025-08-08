"""
Run gamma experiments for 2020 data
Testing gamma values: 0.1, 0.3, 0.5
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

class RunGamma2020:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/gamma_2020_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        # Filter 2020 files
        self.files_2020 = []
        for f in self.all_files:
            if 'test_data_start_date_2020' in f:
                self.files_2020.append(f)
        
        self.files_2020.sort()
        
        print(f"Loaded {len(self.all_files)} total files")
        print(f"Found {len(self.files_2020)} files with 2020 test dates")
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
        """Find file for specific month in 2020."""
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
            print(f"    Warning: Only {len(data_seq)} files loaded, need 7")
            return None
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with 750 iterations
        print(f"    Training {year}-{month:02d} (γ={gamma}): ", end='', flush=True)
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
                print(f"{step+1}", end=' ', flush=True)
        
        train_time = time.time() - train_start
        print(f"Done! [{train_time:.1f}s]")
        
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
            net_return = raw_return - 0.0015  # Transaction cost
            
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = net_return / (torch.std(daily_returns) + 1e-10)
            volatility = torch.std(daily_returns)
            
            return {
                'return': net_return.item(),
                'sharpe': sharpe.item(),
                'raw_return': raw_return.item(),
                'volatility': volatility.item()
            }
    
    def run_gamma_2020(self, gamma):
        """Run 2020 with specific gamma value."""
        
        print(f"\n{'='*70}")
        print(f"YEAR 2020 - GAMMA={gamma} (750 iterations)")
        print(f"{'='*70}")
        
        year_start = time.time()
        
        results = {
            'year': 2020,
            'gamma': gamma,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        # We can only start from month 7 (July) since we need 7 files for training
        # and the earliest test date is April 2020
        for month in range(7, 13):  # July to December
            print(f"\n  Month {month:02d}/2020:")
            
            file_idx, test_file = self.find_month_file(2020, month)
            
            if file_idx is None or file_idx < 6:
                print(f"    Skipping: Insufficient training data")
                continue
            
            # Get 7 training files
            training_files = self.all_files[file_idx-6:file_idx+1]
            test_date = self.get_date(test_file)
            print(f"    Test date: {test_date}")
            
            # Train model
            model = self.train_model_gamma(training_files, gamma, 2020, month)
            
            if model is None:
                print(f"    Training failed")
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
                    'raw_return': result['raw_return'],
                    'volatility': result['volatility'],
                    'ytd': results['cumulative'] - 1
                })
                
                print(f"    Return: {ret*100:+.2f}% | Sharpe: {result['sharpe']:.2f} | YTD: {(results['cumulative']-1)*100:+.2f}%")
            else:
                print(f"    Evaluation failed")
            
            # Clean memory after each month
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate statistics
        if results['monthly_trades']:
            returns = [t['return'] for t in results['monthly_trades']]
            results['annual_return'] = results['cumulative'] - 1
            results['avg_return'] = np.mean(returns)
            results['std_return'] = np.std(returns)
            results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            results['trades'] = len(returns)
            results['best_month'] = max(returns)
            results['worst_month'] = min(returns)
            results['sharpe_ratio'] = results['avg_return'] / (results['std_return'] + 1e-10)
            
            year_time = time.time() - year_start
            
            print(f"\n  2020 SUMMARY (γ={gamma}):")
            print(f"  " + "-"*50)
            print(f"    Annual Return: {results['annual_return']*100:+.2f}%")
            print(f"    Trades: {results['trades']}/6 possible")
            print(f"    Win Rate: {results['win_rate']*100:.0f}%")
            print(f"    Avg Monthly: {results['avg_return']*100:+.2f}%")
            print(f"    Best Month: {results['best_month']*100:+.2f}%")
            print(f"    Worst Month: {results['worst_month']*100:+.2f}%")
            print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"    Processing Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all(self):
        """Run all gamma values for 2020."""
        
        print("\n" + "="*80)
        print("RUNNING GAMMA EXPERIMENTS FOR 2020")
        print("Testing: γ=0.1, γ=0.3, γ=0.5")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        all_results = {}
        
        # Run for each gamma value
        for gamma in [0.1, 0.3, 0.5]:
            results = self.run_gamma_2020(gamma)
            all_results[f'gamma_{gamma}'] = results
            
            # Save individual results
            with open(f'{self.output_dir}gamma{gamma}_2020_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=float)
            
            print(f"\n  Saved γ={gamma} results to: {self.output_dir}gamma{gamma}_2020_results.json")
        
        # Save combined results
        with open(f'{self.output_dir}all_gamma_2020_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        # Print final comparison
        self.print_comparison(all_results)
        
        total_time = time.time() - self.start_time
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
        
        return all_results
    
    def print_comparison(self, results):
        """Print comparison of all gamma values for 2020."""
        
        print("\n" + "="*80)
        print("2020 GAMMA COMPARISON - COMPLETE RESULTS")
        print("="*80)
        
        print(f"\n{'Gamma':<10} | {'Annual Return':>15} | {'Win Rate':>10} | {'Best Month':>12} | {'Worst Month':>12}")
        print("-"*80)
        
        best_return = -999
        best_gamma = None
        
        for gamma in [0.1, 0.3, 0.5]:
            gamma_key = f'gamma_{gamma}'
            if gamma_key in results and 'annual_return' in results[gamma_key]:
                data = results[gamma_key]
                annual_ret = data['annual_return']
                win_rate = data.get('win_rate', 0)
                best_month = data.get('best_month', 0)
                worst_month = data.get('worst_month', 0)
                
                print(f"γ={gamma:<7} | {annual_ret*100:>14.2f}% | {win_rate*100:>9.0f}% | "
                      f"{best_month*100:>11.2f}% | {worst_month*100:>11.2f}%")
                
                if annual_ret > best_return:
                    best_return = annual_ret
                    best_gamma = gamma
        
        print("-"*80)
        
        if best_gamma is not None:
            print(f"\nBEST PERFORMER: γ={best_gamma} with {best_return*100:+.2f}% annual return")
        
        print("\nNOTE: 2020 data only available from July-December (6 months)")
        print("This was a volatile COVID recovery period")


def main():
    """Run gamma experiments for 2020."""
    
    print("Starting gamma experiments for 2020...")
    print("Expected runtime: ~10-15 minutes")
    print("-"*80)
    
    runner = RunGamma2020()
    results = runner.run_all()
    
    print("\n" + "="*60)
    print("2020 GAMMA EXPERIMENTS COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()