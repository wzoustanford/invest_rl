"""
Complete gamma=0.5 experiments for 2023 and 2024 only
Focused script to finish the remaining experiments
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

class CompleteGamma05:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/gamma05_2023_2024_{timestamp}/'
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
            print(f"    Warning: Only {len(data_seq)} files loaded, need 7")
            return None
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with 750 iterations
        print(f"    Training {year}-{month:02d} (γ=0.5): ", end='', flush=True)
        train_start = time.time()
        
        model.train()
        gamma = 0.5  # Fixed gamma value
        
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
    
    def run_year(self, year):
        """Run complete year with gamma=0.5."""
        
        print(f"\n{'='*70}")
        print(f"YEAR {year} - GAMMA=0.5 (750 iterations)")
        print(f"{'='*70}")
        
        year_start = time.time()
        
        results = {
            'year': year,
            'gamma': 0.5,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            print(f"\n  Month {month:02d}/{year}:")
            
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                print(f"    Skipping: Insufficient data")
                continue
            
            # Get 7 training files
            training_files = self.all_files[file_idx-6:file_idx+1]
            test_date = self.get_date(test_file)
            print(f"    Test date: {test_date}")
            
            # Train model
            model = self.train_model_gamma05(training_files, year, month)
            
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
            
            print(f"\n  {year} SUMMARY:")
            print(f"  " + "-"*50)
            print(f"    Annual Return: {results['annual_return']*100:+.2f}%")
            print(f"    Trades: {results['trades']}/12")
            print(f"    Win Rate: {results['win_rate']*100:.0f}%")
            print(f"    Avg Monthly: {results['avg_return']*100:+.2f}%")
            print(f"    Best Month: {results['best_month']*100:+.2f}%")
            print(f"    Worst Month: {results['worst_month']*100:+.2f}%")
            print(f"    Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"    Processing Time: {year_time/60:.1f} minutes")
        
        return results
    
    def run_all(self):
        """Run gamma=0.5 for 2023 and 2024."""
        
        print("\n" + "="*80)
        print("COMPLETING GAMMA=0.5 EXPERIMENTS")
        print("Running: 2023 and 2024 with 750 iterations")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        all_results = {}
        
        # Run 2023 and 2024
        for year in [2023, 2024]:
            year_results = self.run_year(year)
            all_results[year] = year_results
            
            # Save individual year results
            with open(f'{self.output_dir}gamma05_{year}_results.json', 'w') as f:
                json.dump(year_results, f, indent=2, default=float)
            
            print(f"\n  Saved {year} results to: {self.output_dir}gamma05_{year}_results.json")
        
        # Save combined results
        with open(f'{self.output_dir}gamma05_2023_2024_complete.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        # Print final comparison
        self.print_final_comparison(all_results)
        
        total_time = time.time() - self.start_time
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")
        
        return all_results
    
    def print_final_comparison(self, results):
        """Print comparison with other gamma values."""
        
        print("\n" + "="*80)
        print("GAMMA=0.5 COMPLETE RESULTS COMPARISON")
        print("="*80)
        
        # Known results from previous experiments
        known_results = {
            'gamma_0.1': {
                2023: 3.7041,  # +370.41%
                2024: -0.3020  # -30.20%
            },
            'gamma_0.3': {
                2023: 4.0617,  # +406.17%
                2024: -0.3047  # -30.47%
            },
            'gamma_0.5': {
                2021: -0.0733,  # -7.33% (already completed)
                2022: 0.2798   # +27.98% (partial, 10 months)
            }
        }
        
        print(f"\n{'Year':<10} | {'γ=0.1':>15} | {'γ=0.3':>15} | {'γ=0.5':>15} | {'Best':>10}")
        print("-"*80)
        
        # 2023 results
        if 2023 in results and 'annual_return' in results[2023]:
            g05_2023 = results[2023]['annual_return']
            known_results['gamma_0.5'][2023] = g05_2023
            
            returns_2023 = {
                '0.1': known_results['gamma_0.1'][2023],
                '0.3': known_results['gamma_0.3'][2023],
                '0.5': g05_2023
            }
            best_2023 = max(returns_2023, key=returns_2023.get)
            
            print(f"{'2023':<10} | {known_results['gamma_0.1'][2023]*100:>14.2f}% | "
                  f"{known_results['gamma_0.3'][2023]*100:>14.2f}% | "
                  f"{g05_2023*100:>14.2f}% | γ={best_2023:>8}")
        
        # 2024 results
        if 2024 in results and 'annual_return' in results[2024]:
            g05_2024 = results[2024]['annual_return']
            known_results['gamma_0.5'][2024] = g05_2024
            
            returns_2024 = {
                '0.1': known_results['gamma_0.1'][2024],
                '0.3': known_results['gamma_0.3'][2024],
                '0.5': g05_2024
            }
            best_2024 = max(returns_2024, key=returns_2024.get)
            
            print(f"{'2024 (5mo)':<10} | {known_results['gamma_0.1'][2024]*100:>14.2f}% | "
                  f"{known_results['gamma_0.3'][2024]*100:>14.2f}% | "
                  f"{g05_2024*100:>14.2f}% | γ={best_2024:>8}")
        
        print("-"*80)
        
        # Print key insights
        print("\nKEY INSIGHTS:")
        if 2023 in results and 'annual_return' in results[2023]:
            print(f"  2023: Gamma=0.5 achieved {results[2023]['annual_return']*100:+.2f}%")
            if results[2023]['annual_return'] > 3.7:
                print("    -> Second best performance after γ=0.3!")
            
        if 2024 in results and 'annual_return' in results[2024]:
            print(f"  2024: Gamma=0.5 achieved {results[2024]['annual_return']*100:+.2f}% (5 months)")
        
        # Monthly details for interesting patterns
        print("\nNOTABLE MONTHLY RETURNS (γ=0.5):")
        for year in [2023, 2024]:
            if year in results and 'monthly_trades' in results[year]:
                print(f"\n  {year}:")
                for trade in results[year]['monthly_trades']:
                    if abs(trade['return']) > 0.1:  # Show returns > 10%
                        print(f"    Month {trade['month']:02d}: {trade['return']*100:+.2f}%")


def main():
    """Complete gamma=0.5 experiments for 2023 and 2024."""
    
    print("Starting gamma=0.5 completion for 2023 and 2024...")
    print("Expected runtime: ~30-40 minutes")
    print("No timeout will be applied - let it run to completion")
    print("-"*80)
    
    runner = CompleteGamma05()
    results = runner.run_all()
    
    print("\n" + "="*60)
    print("GAMMA=0.5 EXPERIMENTS COMPLETE!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    results = main()