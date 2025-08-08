"""
Complete gamma experiments - continue from where we left off
Run gamma=0.5 for all years and complete gamma=0.1 for 2024
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

class CompleteGammaExperiments:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/gamma_complete_{timestamp}/'
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
        print(f"    Training {year}-{month:02d} (γ={gamma}): ", end='', flush=True)
        
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
            
            raw_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            net_return = raw_return - 0.0015
            
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = net_return / (torch.std(daily_returns) + 1e-10)
            
            return {
                'return': net_return.item(),
                'sharpe': sharpe.item()
            }
    
    def run_year_gamma(self, year, gamma):
        """Run year with specific gamma value."""
        
        print(f"\n  Year {year} - Gamma={gamma}")
        print(f"  " + "-"*60)
        
        results = {
            'year': year,
            'gamma': gamma,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                continue
            
            # Get 7 training files
            training_files = self.all_files[file_idx-6:file_idx+1]
            
            # Train
            model = self.train_model_gamma(training_files, gamma, year, month)
            
            if model is None:
                continue
            
            # Evaluate
            result = self.evaluate_model(model, test_file)
            
            if result:
                ret = result['return']
                results['cumulative'] *= (1 + ret)
                results['monthly_trades'].append({
                    'month': month,
                    'return': ret,
                    'sharpe': result['sharpe'],
                    'ytd': results['cumulative'] - 1
                })
                
                print(f"    M{month:02d}: {ret*100:+.2f}% | YTD: {(results['cumulative']-1)*100:+.2f}%")
            
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
            
            print(f"\n  {year} Annual Return: {results['annual_return']*100:+.2f}%")
            print(f"  Win Rate: {results['win_rate']*100:.0f}%")
        
        return results
    
    def run_all(self):
        """Run remaining experiments."""
        
        print("\n" + "="*80)
        print("COMPLETING GAMMA EXPERIMENTS - 750 ITERATIONS")
        print("="*80)
        
        all_results = {
            'gamma_0.1': {},
            'gamma_0.5': {}
        }
        
        # First, complete gamma=0.1 for 2024
        print("\n" + "="*70)
        print("COMPLETING GAMMA = 0.1 for 2024")
        print("="*70)
        
        # Load existing gamma 0.1 results
        existing_01_dir = '/home/ubuntu/code/angle_rl/invest/experiments/gamma_750iter_20250807_153512/'
        for year in [2021, 2022, 2023]:
            try:
                with open(f'{existing_01_dir}gamma0.1_{year}_results.json', 'r') as f:
                    all_results['gamma_0.1'][year] = json.load(f)
                    print(f"  Loaded existing {year} results for gamma=0.1")
            except:
                print(f"  Running {year} for gamma=0.1")
                all_results['gamma_0.1'][year] = self.run_year_gamma(year, 0.1)
        
        # Complete 2024 for gamma=0.1
        print("\n  Completing 2024 for gamma=0.1")
        all_results['gamma_0.1'][2024] = self.run_year_gamma(2024, 0.1)
        
        # Now run all years for gamma=0.5
        print("\n" + "="*70)
        print("RUNNING GAMMA = 0.5 for all years")
        print("="*70)
        
        for year in [2021, 2022, 2023, 2024]:
            all_results['gamma_0.5'][year] = self.run_year_gamma(year, 0.5)
            
            # Save individual results
            with open(f'{self.output_dir}gamma0.5_{year}_results.json', 'w') as f:
                json.dump(all_results['gamma_0.5'][year], f, indent=2, default=float)
        
        # Save all results
        with open(f'{self.output_dir}all_gamma_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        # Print comprehensive comparison
        self.print_full_comparison(all_results)
        
        return all_results
    
    def print_full_comparison(self, results):
        """Print comprehensive comparison of all gamma values."""
        
        print("\n" + "="*80)
        print("FULL GAMMA COMPARISON - 750 ITERATIONS")
        print("="*80)
        
        # Previous results for gamma=0.3 (baseline)
        baseline_gamma03 = {
            2021: -0.1622,
            2022: 0.1400,
            2023: 4.0617,
            2024: -0.3047
        }
        
        print(f"\n{'Year':<6} | {'γ=0.1':>15} | {'γ=0.3 (base)':>15} | {'γ=0.5':>15} | {'Best γ':>10}")
        print("-"*80)
        
        for year in [2021, 2022, 2023, 2024]:
            base = baseline_gamma03.get(year, 0)
            
            # Get results for each gamma
            g01_data = results.get('gamma_0.1', {}).get(year, {})
            g05_data = results.get('gamma_0.5', {}).get(year, {})
            
            g01 = g01_data.get('annual_return', 0) if isinstance(g01_data, dict) else 0
            g05 = g05_data.get('annual_return', 0) if isinstance(g05_data, dict) else 0
            
            # Find best
            returns = {'0.1': g01, '0.3': base, '0.5': g05}
            best_gamma = max(returns, key=returns.get)
            
            print(f"{year:<6} | {g01*100:>14.2f}% | {base*100:>14.2f}% | {g05*100:>14.2f}% | γ={best_gamma:>8}")
        
        print("-"*80)
        
        # Calculate averages for 2021-2023
        years_for_avg = [2021, 2022, 2023]
        
        avg_g01 = np.mean([results.get('gamma_0.1', {}).get(y, {}).get('annual_return', 0) 
                          for y in years_for_avg])
        avg_g03 = np.mean([baseline_gamma03[y] for y in years_for_avg])
        avg_g05 = np.mean([results.get('gamma_0.5', {}).get(y, {}).get('annual_return', 0) 
                          for y in years_for_avg])
        
        print(f"{'3-Yr Avg':<6} | {avg_g01*100:>14.2f}% | {avg_g03*100:>14.2f}% | {avg_g05*100:>14.2f}% |")
        print("-"*80)
        
        # Additional statistics
        print("\n" + "="*80)
        print("DETAILED STATISTICS BY GAMMA")
        print("="*80)
        
        for gamma in [0.1, 0.5]:
            print(f"\nGamma = {gamma}:")
            print("-"*40)
            
            gamma_key = f'gamma_{gamma}'
            if gamma_key in results:
                for year in [2021, 2022, 2023, 2024]:
                    if year in results[gamma_key]:
                        year_data = results[gamma_key][year]
                        if isinstance(year_data, dict) and 'annual_return' in year_data:
                            print(f"  {year}:")
                            print(f"    Annual Return: {year_data['annual_return']*100:+.2f}%")
                            print(f"    Win Rate: {year_data.get('win_rate', 0)*100:.0f}%")
                            print(f"    Best Month: {year_data.get('best_month', 0)*100:+.2f}%")
                            print(f"    Worst Month: {year_data.get('worst_month', 0)*100:+.2f}%")
                            print(f"    Trades: {year_data.get('trades', 0)}/12")
        
        print("\n" + "="*80)
        print("KEY INSIGHTS")
        print("="*80)
        
        print("\n1. Gamma Impact on Performance:")
        print("   - γ=0.1 (less discounting): Weights historical data more equally")
        print("   - γ=0.3 (balanced): Optimal for most market conditions")
        print("   - γ=0.5 (more discounting): Heavy emphasis on most recent data")
        
        print("\n2. Best Configuration by Year:")
        for year in [2021, 2022, 2023, 2024]:
            returns = {
                '0.1': results.get('gamma_0.1', {}).get(year, {}).get('annual_return', -999),
                '0.3': baseline_gamma03.get(year, -999),
                '0.5': results.get('gamma_0.5', {}).get(year, {}).get('annual_return', -999)
            }
            best = max(returns, key=returns.get)
            best_return = returns[best]
            if best_return > -999:
                print(f"   {year}: γ={best} with {best_return*100:+.2f}%")
        
        total_time = time.time() - self.start_time
        print(f"\nTotal execution time: {total_time/60:.1f} minutes")
        print(f"Results saved to: {self.output_dir}")


def main():
    """Complete gamma experiments."""
    
    runner = CompleteGammaExperiments()
    results = runner.run_all()
    
    return results


if __name__ == "__main__":
    results = main()