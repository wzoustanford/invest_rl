"""
Run Sequential Supervised Learning with 750 iterations
Testing on 2021, 2022, 2023, 2024 data
Step 1 of model improvements
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Optional

from model.iimodel import IIMODEL

class SequentialTrainer750:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/750iter_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files")
        print(f"Output directory: {self.output_dir}")
    
    def get_date(self, filename: str) -> Optional[str]:
        """Extract date from filename."""
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    def find_month_file(self, year: int, month: int) -> tuple:
        """Find file index for specific year and month."""
        for i, filepath in enumerate(self.all_files):
            date_str = self.get_date(filepath)
            if date_str:
                file_year = int(date_str[:4])
                file_month = int(date_str[5:7])
                if file_year == year and file_month == month:
                    return i, filepath
        return None, None
    
    def train_model(self, training_files: List[str], iterations: int = 750) -> Optional[torch.nn.Module]:
        """Train model with 750 iterations."""
        
        # Load 7 consecutive files
        data_seq = []
        for f in training_files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    data_seq.append(pickle.load(file))
        
        if len(data_seq) != 7:
            return None
        
        # Initialize model
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training with 750 iterations
        print(f"      Training with {iterations} iterations...", end='')
        for step in range(iterations):
            model.train()
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0).to(self.device)
            
            # Process each of 7 files with gamma discounting
            for i in range(7):
                features = data_seq[i]['trainFeature'].to(self.device)
                series = data_seq[i]['train_in_portfolio_series'].to(self.device)
                
                # Get portfolio weights
                weights = model(features)
                
                # Calculate return (day 25 - day 1)
                shares = weights / (series[:, 0:1] + 1e-10)
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                # Calculate Sharpe ratio
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                # Gamma discounting (gamma = 0.3)
                gamma_power = 0.3 ** (7 - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
            
            # Progress indicator
            if (step + 1) % 150 == 0:
                print(f" {step+1}/{iterations}", end='')
        
        print(" Done!")
        return model
    
    def evaluate_model(self, model: torch.nn.Module, test_file: str) -> Optional[Dict]:
        """Evaluate model on test data."""
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
            ret = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            
            # Apply transaction cost
            ret = ret - 0.0015
            
            # Calculate Sharpe
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = ret / (torch.std(daily_returns) + 1e-10)
            
            return {'return': ret.item(), 'sharpe': sharpe.item()}
    
    def run_year(self, year: int, iterations: int = 750) -> Dict:
        """Run 12 monthly trades for a year."""
        
        print(f"\n  Year {year} (750 iterations):")
        print("  " + "-" * 60)
        
        results = {
            'year': year,
            'iterations': iterations,
            'monthly_trades': [],
            'cumulative': 1.0
        }
        
        trades_executed = 0
        
        for month in range(1, 13):
            # Find file for this month
            file_idx, test_file = self.find_month_file(year, month)
            
            if file_idx is None or file_idx < 6:
                print(f"    Month {month:02d}: No data or insufficient history")
                continue
            
            # Get 7 training files
            training_files = self.all_files[file_idx-6:file_idx+1]
            
            test_date = self.get_date(test_file)
            print(f"    Month {month:02d} ({test_date}): ", end='')
            
            # Train model with 750 iterations
            model = self.train_model(training_files, iterations=iterations)
            
            if model is None:
                print("Training failed")
                continue
            
            # Evaluate
            result = self.evaluate_model(model, test_file)
            
            if result is not None:
                ret = result['return']
                sharpe = result['sharpe']
                results['cumulative'] *= (1 + ret)
                results['monthly_trades'].append({
                    'month': month,
                    'date': test_date,
                    'file_idx': file_idx,
                    'return': ret,
                    'sharpe': sharpe,
                    'cumulative_ytd': results['cumulative'] - 1
                })
                trades_executed += 1
                
                print(f"Return: {ret*100:+.2f}%, YTD: {(results['cumulative']-1)*100:+.2f}%")
            else:
                print("Evaluation failed")
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
        
        # Calculate statistics
        if results['monthly_trades']:
            returns = [t['return'] for t in results['monthly_trades']]
            results['statistics'] = {
                'annual_return': results['cumulative'] - 1,
                'trades_executed': trades_executed,
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-10),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                'best_month': max(returns),
                'worst_month': min(returns)
            }
        
        return results
    
    def run_all_years(self):
        """Run experiment for all years."""
        
        print("\n" + "="*80)
        print("SEQUENTIAL SUPERVISED LEARNING - 750 ITERATIONS")
        print("Testing Model Improvement Step 1: Increased Training Iterations")
        print("="*80)
        
        all_results = {}
        
        # Run for 2021, 2022, 2023, 2024
        for year in [2021, 2022, 2023, 2024]:
            print(f"\n{'='*80}")
            print(f"RUNNING YEAR {year}")
            print(f"{'='*80}")
            
            year_results = self.run_year(year, iterations=750)
            all_results[year] = year_results
            
            # Save individual year results
            with open(f'{self.output_dir}{year}_results.json', 'w') as f:
                json.dump(year_results, f, indent=2, default=float)
            
            # Print year summary
            if 'statistics' in year_results:
                stats = year_results['statistics']
                print(f"\n  {year} Summary:")
                print(f"    Annual Return: {stats['annual_return']*100:+.2f}%")
                print(f"    Trades: {stats['trades_executed']}/12")
                print(f"    Win Rate: {stats['win_rate']*100:.0f}%")
                print(f"    Avg per Trade: {stats['avg_return']*100:+.2f}%")
        
        # Save all results
        with open(f'{self.output_dir}all_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=float)
        
        return all_results
    
    def print_comparison(self, results: Dict):
        """Print comparison with baseline (100 iterations)."""
        
        print("\n" + "="*80)
        print("COMPARISON: 750 ITERATIONS vs BASELINE (100 iterations)")
        print("="*80)
        
        # Baseline results from previous experiments
        baseline = {
            2021: -0.3379,  # From previous 2021 experiment
            2022: 0.1387,   # From previous experiments
            2023: 0.1088,   # From previous experiments
            2024: None      # Not yet run
        }
        
        print("\n" + "-"*80)
        print(f"{'Year':<10} | {'Baseline (100 iter)':>20} | {'750 Iterations':>20} | {'Improvement':>15}")
        print("-"*80)
        
        for year in [2021, 2022, 2023, 2024]:
            if year in results and 'statistics' in results[year]:
                new_return = results[year]['statistics']['annual_return']
                base_return = baseline.get(year)
                
                if base_return is not None:
                    improvement = (new_return - base_return) * 100
                    print(f"{year:<10} | {base_return*100:>19.2f}% | {new_return*100:>19.2f}% | {improvement:+14.2f}pp")
                else:
                    print(f"{year:<10} | {'N/A':>20} | {new_return*100:>19.2f}% | {'N/A':>15}")
        
        print("-"*80)
        
        # Calculate averages
        years_with_baseline = [y for y in [2021, 2022, 2023] if y in results and 'statistics' in results[y]]
        if years_with_baseline:
            avg_baseline = np.mean([baseline[y] for y in years_with_baseline])
            avg_750 = np.mean([results[y]['statistics']['annual_return'] for y in years_with_baseline])
            avg_improvement = (avg_750 - avg_baseline) * 100
            
            print(f"{'3-Yr Avg':<10} | {avg_baseline*100:>19.2f}% | {avg_750*100:>19.2f}% | {avg_improvement:+14.2f}pp")
            print("-"*80)
        
        # Summary statistics
        print("\nKey Insights:")
        if avg_improvement > 0:
            print(f"✓ 750 iterations improved performance by {avg_improvement:.2f} percentage points on average")
        else:
            print(f"✗ 750 iterations decreased performance by {abs(avg_improvement):.2f} percentage points on average")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """Run 750 iteration experiment."""
    
    trainer = SequentialTrainer750()
    results = trainer.run_all_years()
    trainer.print_comparison(results)
    
    return results


if __name__ == "__main__":
    import time
    start_time = time.time()
    
    results = main()
    
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed/60:.1f} minutes")