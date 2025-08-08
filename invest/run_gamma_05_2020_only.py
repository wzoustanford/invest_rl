"""
Complete gamma=0.5 for 2020 only
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

class RunGamma05_2020:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        self.output_dir = '/home/ubuntu/code/angle_rl/invest/experiments/gamma_2020_20250807_224216/'
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} total files")
        print(f"Device: {self.device}")
        print("="*80)
    
    def get_date(self, filename):
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    def find_month_file(self, year, month):
        """Find file for specific month."""
        for i, filepath in enumerate(self.all_files):
            date_str = self.get_date(filepath)
            if date_str:
                file_year = int(date_str[:4])
                file_month = int(date_str[5:7])
                if file_year == year and file_month == month:
                    return i, filepath
        return None, None
    
    def train_model_gamma05(self, training_files, year, month):
        """Train with gamma=0.5 and 750 iterations."""
        
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
        print(f"    Training {year}-{month:02d} (γ=0.5): ", end='', flush=True)
        train_start = time.time()
        
        model.train()
        gamma = 0.5
        
        for step in range(750):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Process 7 files with gamma=0.5
            for i in range(7):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                # Apply gamma=0.5 discounting
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
            net_return = raw_return - 0.0015
            
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
    
    def run_2020_gamma05(self):
        """Run 2020 with gamma=0.5."""
        
        print("\n" + "="*70)
        print("YEAR 2020 - GAMMA=0.5 (750 iterations)")
        print("="*70)
        
        year_start = time.time()
        
        results = {
            'year': 2020,
            'gamma': 0.5,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        # July to December
        for month in range(7, 13):
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
            model = self.train_model_gamma05(training_files, 2020, month)
            
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
            
            # Clean memory
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
            
            print(f"\n  2020 SUMMARY (γ=0.5):")
            print(f"  " + "-"*50)
            print(f"    Annual Return: {results['annual_return']*100:+.2f}%")
            print(f"    Trades: {results['trades']}/6 possible")
            print(f"    Win Rate: {results['win_rate']*100:.0f}%")
            print(f"    Avg Monthly: {results['avg_return']*100:+.2f}%")
            print(f"    Best Month: {results['best_month']*100:+.2f}%")
            print(f"    Worst Month: {results['worst_month']*100:+.2f}%")
            print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"    Processing Time: {year_time/60:.1f} minutes")
        
        # Save results
        with open(f'{self.output_dir}gamma0.5_2020_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=float)
        
        print(f"\n  Saved γ=0.5 results to: {self.output_dir}gamma0.5_2020_results.json")
        
        return results


def main():
    """Complete gamma=0.5 for 2020."""
    
    print("Completing gamma=0.5 experiment for 2020...")
    print("Expected runtime: ~10 minutes")
    print("-"*80)
    
    runner = RunGamma05_2020()
    results = runner.run_2020_gamma05()
    
    print("\n" + "="*60)
    print("2020 GAMMA=0.5 EXPERIMENT COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()