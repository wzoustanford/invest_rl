"""
Optimized 750 iteration experiment with better memory management and parallel processing
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime
import gc

from model.iimodel import IIMODEL

# Enable mixed precision for faster training
torch.backends.cudnn.benchmark = True

class FastSequentialTrainer750:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/750iter_opt_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
    
    def get_date(self, filename: str):
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    def find_month_file(self, year: int, month: int):
        for i, filepath in enumerate(self.all_files):
            date_str = self.get_date(filepath)
            if date_str:
                file_year = int(date_str[:4])
                file_month = int(date_str[5:7])
                if file_year == year and file_month == month:
                    return i, filepath
        return None, None
    
    def train_model_fast(self, training_files, iterations=750):
        """Optimized training with gradient accumulation."""
        
        # Load data once
        data_seq = []
        for f in training_files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    data = pickle.load(file)
                    # Pre-move to device
                    data['trainFeature'] = data['trainFeature'].to(self.device)
                    data['train_in_portfolio_series'] = data['train_in_portfolio_series'].to(self.device)
                    data_seq.append(data)
        
        if len(data_seq) != 7:
            return None
        
        # Model setup
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Use gradient accumulation for better memory efficiency
        accumulation_steps = 10
        
        # Training loop
        model.train()
        for step in range(iterations):
            if step % accumulation_steps == 0:
                optimizer.zero_grad()
            
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Process files
            for i in range(7):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                gamma_power = 0.3 ** (7 - i - 1)
                loss = -sharpe * gamma_power / accumulation_steps
                total_loss = total_loss + loss
            
            total_loss.backward()
            
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
            
            # Progress
            if (step + 1) % 250 == 0:
                print(f".", end='', flush=True)
        
        return model
    
    def evaluate_model(self, model, test_file):
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
    
    def run_year_fast(self, year: int):
        """Run year with optimized memory management."""
        
        print(f"\nYear {year}:")
        
        results = {
            'year': year,
            'iterations': 750,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                continue
            
            training_files = self.all_files[file_idx-6:file_idx+1]
            test_date = self.get_date(test_file)
            
            print(f"  M{month:02d}: ", end='')
            
            # Train
            model = self.train_model_fast(training_files, iterations=750)
            
            if model is None:
                print("Failed")
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
                    'sharpe': result['sharpe']
                })
                print(f" {ret*100:+.2f}%")
            else:
                print("Eval failed")
            
            # Aggressive memory cleanup
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Statistics
        if results['monthly_trades']:
            returns = [t['return'] for t in results['monthly_trades']]
            results['annual_return'] = results['cumulative'] - 1
            results['avg_return'] = np.mean(returns)
            results['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
            results['trades'] = len(returns)
        
        return results
    
    def run_all(self):
        """Run all years."""
        
        print("\n" + "="*70)
        print("750 ITERATION EXPERIMENT - OPTIMIZED")
        print("="*70)
        
        all_results = {}
        
        for year in [2021, 2022, 2023, 2024]:
            year_results = self.run_year_fast(year)
            all_results[year] = year_results
            
            if 'annual_return' in year_results:
                print(f"  {year} Annual: {year_results['annual_return']*100:+.2f}% ({year_results['trades']} trades)")
        
        # Save results
        with open(f'{self.output_dir}results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        # Print comparison
        print("\n" + "="*70)
        print("COMPARISON WITH BASELINE")
        print("="*70)
        
        baseline = {2021: -0.3379, 2022: 0.1387, 2023: 0.1088}
        
        print(f"\n{'Year':<6} | {'Baseline (100 iter)':>20} | {'750 iter':>20} | {'Diff':>10}")
        print("-"*70)
        
        improvements = []
        for year in [2021, 2022, 2023]:
            if year in all_results and 'annual_return' in all_results[year]:
                new_ret = all_results[year]['annual_return']
                base_ret = baseline[year]
                diff = (new_ret - base_ret) * 100
                improvements.append(diff)
                print(f"{year:<6} | {base_ret*100:>19.2f}% | {new_ret*100:>19.2f}% | {diff:+9.2f}pp")
        
        if improvements:
            avg_improvement = np.mean(improvements)
            print("-"*70)
            print(f"Average improvement: {avg_improvement:+.2f} percentage points")
        
        # Year 2024 (no baseline)
        if 2024 in all_results and 'annual_return' in all_results[2024]:
            print(f"\n2024 (new): {all_results[2024]['annual_return']*100:+.2f}%")
        
        print(f"\nResults saved to: {self.output_dir}")
        
        return all_results


def main():
    trainer = FastSequentialTrainer750()
    return trainer.run_all()


if __name__ == "__main__":
    import time
    start = time.time()
    results = main()
    print(f"\nCompleted in {(time.time()-start)/60:.1f} minutes")