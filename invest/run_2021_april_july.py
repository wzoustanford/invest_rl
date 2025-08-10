"""
Run sequential supervised learning for April-July 2021
Using 7-day sequences with gamma discounting
Trading starts from files with test_data_start_date_2020_04_XX through 2020_07_XX
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime, timedelta
import time
import gc

from model.iimodel import IIMODEL

class Run2021AprilJuly:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/april_july_2021_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} total files")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print("="*80)
        
        self.start_time = time.time()
    
    def get_date(self, filename):
        """Extract test_data_start_date from filename"""
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    def calculate_actual_trading_date(self, test_date_str):
        """Calculate actual trading date: test_date + 360 days"""
        test_date = datetime.strptime(test_date_str, '%Y-%m-%d')
        trading_start = test_date + timedelta(days=360)
        return trading_start
    
    def find_files_for_month(self, target_year, target_month):
        """Find all files that trade in a specific month"""
        matching_files = []
        for filepath in self.all_files:
            test_date_str = self.get_date(filepath)
            if test_date_str:
                trading_date = self.calculate_actual_trading_date(test_date_str)
                if trading_date.year == target_year and trading_date.month == target_month:
                    matching_files.append((filepath, test_date_str, trading_date))
        return matching_files
    
    def train_model_gamma(self, training_files, gamma, trading_date):
        """Train model with 7-day sequence and gamma discounting"""
        
        # Load 7 consecutive files
        data_seq = []
        for f in training_files:
            if os.path.exists(f):
                try:
                    with open(f, 'rb') as file:
                        data = pickle.load(file)
                        data['trainFeature'] = data['trainFeature'].to(self.device)
                        data['train_in_portfolio_series'] = data['train_in_portfolio_series'].to(self.device)
                        data_seq.append(data)
                except Exception as e:
                    print(f"    Error loading {f}: {e}")
                    return None
        
        if len(data_seq) != 7:
            print(f"    Warning: Only {len(data_seq)} files loaded, need 7")
            return None
        
        # Initialize model
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with 750 iterations
        print(f"    Training (γ={gamma}): ", end='', flush=True)
        train_start = time.time()
        
        model.train()
        for step in range(750):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Process 7-day sequence with gamma discounting
            for i in range(7):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                # Forward pass
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                
                # Calculate portfolio return for 25-day holding period
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                # Calculate Sharpe ratio
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                # Apply gamma discounting: γ^(T-i-1) where T=7
                gamma_power = gamma ** (7 - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            # Progress indicator
            if (step + 1) % 150 == 0:
                print(f"{step+1}", end=' ', flush=True)
        
        train_time = time.time() - train_start
        print(f"Done! [{train_time:.1f}s]")
        
        return model
    
    def evaluate_model(self, model, test_file):
        """Evaluate model on test data"""
        if not os.path.exists(test_file) or model is None:
            return None
        
        try:
            with open(test_file, 'rb') as f:
                test_data = pickle.load(f)
        except Exception as e:
            print(f"    Error loading test file: {e}")
            return None
        
        if test_data.get('test_in_portfolio_series') is None:
            return None
        
        model.eval()
        with torch.no_grad():
            features = test_data['testFeature'].to(self.device)
            series = test_data['test_in_portfolio_series'].to(self.device)
            
            # Get portfolio weights
            weights = model(features)
            shares = weights / (series[:, 0:1] + 1e-10)
            
            # Calculate returns
            raw_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            net_return = raw_return - 0.0015  # 0.15% transaction cost
            
            # Calculate Sharpe ratio
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
    
    def run_april_july_2021(self, gamma):
        """Run April-July 2021 trading with specified gamma"""
        
        print(f"\n{'='*70}")
        print(f"APRIL-JULY 2021 TRADING - GAMMA={gamma}")
        print(f"{'='*70}")
        
        results = {
            'gamma': gamma,
            'iterations': 750,
            'trades': [],
            'cumulative': 1.0
        }
        
        # Process April through July 2021
        for month in [4, 5, 6, 7]:  # April to July
            month_names = {4: 'April', 5: 'May', 6: 'June', 7: 'July'}
            month_name = month_names[month]
            print(f"\n  {month_name} 2021 Trading:")
            
            # Find files that trade in this month
            month_files = self.find_files_for_month(2021, month)
            
            if not month_files:
                print(f"    No files found for {month_name} 2021 trading")
                continue
            
            # Use the first available file for the month
            test_file, test_date_str, trading_date = month_files[0]
            print(f"    Using file: {os.path.basename(test_file)}")
            print(f"    Test date in file: {test_date_str}")
            print(f"    Actual trading starts: {trading_date.date()}")
            
            # Find the index of this file
            file_idx = None
            for i, f in enumerate(self.all_files):
                if f == test_file:
                    file_idx = i
                    break
            
            if file_idx is None or file_idx < 6:
                print(f"    Insufficient training files")
                continue
            
            # Get 7 training files (current + 6 previous)
            training_files = self.all_files[file_idx-6:file_idx+1]
            print(f"    Using {len(training_files)} training files")
            print(f"    First: {self.get_date(training_files[0])}")
            print(f"    Last: {self.get_date(training_files[-1])}")
            
            # Train model
            model = self.train_model_gamma(training_files, gamma, trading_date)
            
            if model is None:
                print(f"    Training failed")
                continue
            
            # Evaluate on test data
            result = self.evaluate_model(model, test_file)
            
            if result:
                ret = result['return']
                results['cumulative'] *= (1 + ret)
                
                trade_info = {
                    'month': month_name,
                    'file_date': test_date_str,
                    'trading_date': trading_date.strftime('%Y-%m-%d'),
                    'return': ret,
                    'sharpe': result['sharpe'],
                    'raw_return': result['raw_return'],
                    'volatility': result['volatility'],
                    'cumulative': results['cumulative']
                }
                results['trades'].append(trade_info)
                
                print(f"    Return: {ret*100:+.2f}% | Sharpe: {result['sharpe']:.2f} | Cumulative: {(results['cumulative']-1)*100:+.2f}%")
            else:
                print(f"    Evaluation failed")
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate summary statistics
        if results['trades']:
            returns = [t['return'] for t in results['trades']]
            results['total_return'] = results['cumulative'] - 1
            results['avg_return'] = np.mean(returns)
            results['std_return'] = np.std(returns)
            results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            results['num_trades'] = len(returns)
            results['best_trade'] = max(returns)
            results['worst_trade'] = min(returns)
            results['sharpe_ratio'] = results['avg_return'] / (results['std_return'] + 1e-10)
            
            print(f"\n  APRIL-JULY 2021 SUMMARY (γ={gamma}):")
            print(f"  " + "-"*50)
            print(f"    Total Return: {results['total_return']*100:+.2f}%")
            print(f"    Trades: {results['num_trades']}/4")
            print(f"    Win Rate: {results['win_rate']*100:.0f}%")
            print(f"    Avg Monthly: {results['avg_return']*100:+.2f}%")
            print(f"    Best Trade: {results['best_trade']*100:+.2f}%")
            print(f"    Worst Trade: {results['worst_trade']*100:+.2f}%")
            print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        
        return results
    
    def run_all_gammas(self):
        """Run all gamma values for April-July 2021"""
        
        print("\n" + "="*80)
        print("RUNNING APRIL-JULY 2021 EXPERIMENTS")
        print("Testing: γ=0.1, γ=0.3, γ=0.5")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        all_results = {}
        
        # Run for each gamma
        for gamma in [0.1, 0.3, 0.5]:
            results = self.run_april_july_2021(gamma)
            all_results[f'gamma_{gamma}'] = results
            
            # Save results
            with open(f'{self.output_dir}gamma{gamma}_april_july_2021.json', 'w') as f:
                json.dump(results, f, indent=2, default=float)
        
        # Print comparison
        self.print_comparison(all_results)
        
        total_time = time.time() - self.start_time
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
        
        return all_results
    
    def print_comparison(self, results):
        """Print comparison of all gamma values"""
        
        print("\n" + "="*80)
        print("APRIL-JULY 2021 COMPARISON")
        print("="*80)
        
        print(f"\n{'Gamma':<10} | {'Total Return':>15} | {'Win Rate':>10} | {'Best Trade':>12} | {'Worst Trade':>12}")
        print("-"*80)
        
        for gamma in [0.1, 0.3, 0.5]:
            key = f'gamma_{gamma}'
            if key in results and 'total_return' in results[key]:
                data = results[key]
                print(f"γ={gamma:<7} | {data['total_return']*100:>14.2f}% | "
                      f"{data.get('win_rate', 0)*100:>9.0f}% | "
                      f"{data.get('best_trade', 0)*100:>11.2f}% | "
                      f"{data.get('worst_trade', 0)*100:>11.2f}%")
        
        print("\nS&P 500 April-July 2021: ~+13.5%")
        print("(Apr: +5.2%, May: +0.5%, Jun: +2.2%, Jul: +2.3%)")


def main():
    """Run April-July 2021 experiments"""
    
    print("="*60)
    print("SEQUENTIAL SUPERVISED LEARNING")
    print("April-July 2021 Trading Experiments")
    print("Gamma values: 0.1, 0.3, 0.5")
    print("="*60)
    
    runner = Run2021AprilJuly()
    results = runner.run_all_gammas()
    
    print("\n" + "="*60)
    print("APRIL-JULY 2021 EXPERIMENTS COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()