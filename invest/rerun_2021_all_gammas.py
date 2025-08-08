"""
Rerun 2021 trading experiments with gamma = 0.1, 0.3, 0.5
Trading from July-December 2021 (using test_data_start_date_2020 files)
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

class Rerun2021AllGammas:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/rerun_2021_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create log file
        self.log_file = open(f'{self.output_dir}experiment_log.txt', 'w')
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        self.log_print(f"Loaded {len(self.all_files)} total files")
        self.log_print(f"Device: {self.device}")
        self.log_print(f"Output: {self.output_dir}")
        self.log_print("="*80)
        
        self.start_time = time.time()
    
    def log_print(self, msg):
        """Print and log message"""
        print(msg)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
    
    def get_date(self, filename):
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    def find_month_file(self, year, month):
        """Find file for specific month in 2020 (for 2021 trading)."""
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
            self.log_print(f"    Warning: Only {len(data_seq)} files loaded, need 7")
            return None
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with 750 iterations
        print(f"    Training 2021-{month:02d} (γ={gamma}): ", end='', flush=True)
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
            
            # Get top stock picks for analysis
            top_weights, top_indices = torch.topk(weights.squeeze(), k=min(10, weights.shape[0]))
            
            return {
                'return': net_return.item(),
                'sharpe': sharpe.item(),
                'raw_return': raw_return.item(),
                'volatility': volatility.item(),
                'top_weights': top_weights.cpu().numpy().tolist(),
                'top_indices': top_indices.cpu().numpy().tolist()
            }
    
    def run_2021_gamma(self, gamma):
        """Run 2021 (H2) with specific gamma value."""
        
        self.log_print(f"\n{'='*70}")
        self.log_print(f"2021 TRADING (H2) - GAMMA={gamma} (750 iterations)")
        self.log_print(f"{'='*70}")
        
        year_start = time.time()
        
        results = {
            'year': 2021,
            'gamma': gamma,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        # July to December 2021 trading (using 2020 test date files)
        for month in range(7, 13):
            self.log_print(f"\n  Month {month:02d}/2021:")
            
            # Find file with test_data_start_date_2020_XX
            file_idx, test_file = self.find_month_file(2020, month)
            
            if file_idx is None or file_idx < 6:
                self.log_print(f"    Skipping: Insufficient training data")
                continue
            
            # Get 7 training files
            training_files = self.all_files[file_idx-6:file_idx+1]
            test_date = self.get_date(test_file)
            self.log_print(f"    Test date in file: {test_date} (actual trading: 2021-{month:02d})")
            
            # Verify we're using the right files
            self.log_print(f"    Using {len(training_files)} training files")
            self.log_print(f"    First file date: {self.get_date(training_files[0])}")
            self.log_print(f"    Last file date: {self.get_date(training_files[-1])}")
            
            # Train model
            model = self.train_model_gamma(training_files, gamma, 2021, month)
            
            if model is None:
                self.log_print(f"    Training failed")
                continue
            
            # Evaluate
            result = self.evaluate_model(model, test_file)
            
            if result:
                ret = result['return']
                results['cumulative'] *= (1 + ret)
                results['monthly_trades'].append({
                    'month': month,
                    'file_date': test_date,
                    'actual_trading_month': f'2021-{month:02d}',
                    'return': ret,
                    'sharpe': result['sharpe'],
                    'raw_return': result['raw_return'],
                    'volatility': result['volatility'],
                    'ytd': results['cumulative'] - 1,
                    'top_weights': result['top_weights'][:5]  # Save top 5 weights
                })
                
                self.log_print(f"    Return: {ret*100:+.2f}% | Sharpe: {result['sharpe']:.2f} | YTD: {(results['cumulative']-1)*100:+.2f}%")
            else:
                self.log_print(f"    Evaluation failed")
            
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
            
            self.log_print(f"\n  2021 H2 SUMMARY (γ={gamma}):")
            self.log_print(f"  " + "-"*50)
            self.log_print(f"    Annual Return: {results['annual_return']*100:+.2f}%")
            self.log_print(f"    Trades: {results['trades']}/6 possible")
            self.log_print(f"    Win Rate: {results['win_rate']*100:.0f}%")
            self.log_print(f"    Avg Monthly: {results['avg_return']*100:+.2f}%")
            self.log_print(f"    Best Month: {results['best_month']*100:+.2f}%")
            self.log_print(f"    Worst Month: {results['worst_month']*100:+.2f}%")
            self.log_print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            self.log_print(f"    Processing Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all(self):
        """Run all gamma values for 2021."""
        
        self.log_print("\n" + "="*80)
        self.log_print("RERUNNING 2021 (H2) TRADING EXPERIMENTS")
        self.log_print("Testing: γ=0.1, γ=0.3, γ=0.5")
        self.log_print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log_print("="*80)
        
        all_results = {}
        
        # Run for each gamma value
        for gamma in [0.1, 0.3, 0.5]:
            results = self.run_2021_gamma(gamma)
            all_results[f'gamma_{gamma}'] = results
            
            # Save individual results
            with open(f'{self.output_dir}gamma{gamma}_2021_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=float)
            
            self.log_print(f"\n  Saved γ={gamma} results to: {self.output_dir}gamma{gamma}_2021_results.json")
        
        # Save combined results
        with open(f'{self.output_dir}all_gamma_2021_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        # Print final comparison
        self.print_comparison(all_results)
        
        total_time = time.time() - self.start_time
        self.log_print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        self.log_print(f"Results saved to: {self.output_dir}")
        
        self.log_file.close()
        
        return all_results
    
    def print_comparison(self, results):
        """Print comparison of all gamma values for 2021."""
        
        self.log_print("\n" + "="*80)
        self.log_print("2021 (H2) GAMMA COMPARISON - RERUN RESULTS")
        self.log_print("="*80)
        
        self.log_print(f"\n{'Gamma':<10} | {'Annual Return':>15} | {'Win Rate':>10} | {'Best Month':>12} | {'Worst Month':>12}")
        self.log_print("-"*80)
        
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
                
                self.log_print(f"γ={gamma:<7} | {annual_ret*100:>14.2f}% | {win_rate*100:>9.0f}% | "
                      f"{best_month*100:>11.2f}% | {worst_month*100:>11.2f}%")
                
                if annual_ret > best_return:
                    best_return = annual_ret
                    best_gamma = gamma
        
        self.log_print("-"*80)
        
        if best_gamma is not None:
            self.log_print(f"\nBEST PERFORMER: γ={best_gamma} with {best_return*100:+.2f}% return")
        
        self.log_print("\nCONTEXT:")
        self.log_print("• S&P 500 in H2 2021: ~+11%")
        self.log_print("• Market: Strong bull market, low volatility")
        self.log_print("• Training data: 2020 COVID crash period")
        
        # Month by month comparison
        self.log_print("\nMONTH BY MONTH COMPARISON:")
        self.log_print("-"*60)
        self.log_print(f"{'Month':<6} | {'γ=0.1':>10} | {'γ=0.3':>10} | {'γ=0.5':>10}")
        self.log_print("-"*60)
        
        months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for i, month in enumerate(months):
            g01_ret = results.get('gamma_0.1', {}).get('monthly_trades', [{}])[i].get('return', 0) * 100 if i < len(results.get('gamma_0.1', {}).get('monthly_trades', [])) else 0
            g03_ret = results.get('gamma_0.3', {}).get('monthly_trades', [{}])[i].get('return', 0) * 100 if i < len(results.get('gamma_0.3', {}).get('monthly_trades', [])) else 0
            g05_ret = results.get('gamma_0.5', {}).get('monthly_trades', [{}])[i].get('return', 0) * 100 if i < len(results.get('gamma_0.5', {}).get('monthly_trades', [])) else 0
            
            self.log_print(f"{month:<6} | {g01_ret:>9.2f}% | {g03_ret:>9.2f}% | {g05_ret:>9.2f}%")


def main():
    """Rerun 2021 gamma experiments."""
    
    print("="*60)
    print("RERUNNING 2021 (H2) TRADING EXPERIMENTS")
    print("Gamma values: 0.1, 0.3, 0.5")
    print("Expected runtime: ~20-25 minutes")
    print("="*60)
    
    runner = Rerun2021AllGammas()
    results = runner.run_all()
    
    print("\n" + "="*60)
    print("2021 RERUN EXPERIMENTS COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()