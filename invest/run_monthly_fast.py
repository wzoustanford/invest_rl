"""
Fast monthly trading experiment - optimized for speed
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple

from model.iimodel import IIMODEL


def run_fast_monthly_experiment():
    """Run a fast version of monthly trading with reduced training steps."""
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(42)
    np.random.seed(42)
    
    data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'{data_dir}seq_monthly_fast_{timestamp}/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data files
    with open(f'{data_dir}all_data_list.txt', 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(all_files)} data files")
    
    # Extract date from filename
    def get_date(filename):
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    # Find files for each month
    def find_month_files(year, month):
        for i, filepath in enumerate(all_files):
            date_str = get_date(filepath)
            if date_str:
                file_year = int(date_str[:4])
                file_month = int(date_str[5:7])
                if file_year == year and file_month == month:
                    # Get 7 training files
                    if i >= 6:
                        return all_files[i-6:i+1], all_files[i]
        return None, None
    
    # Quick training function
    def quick_train(train_files, steps=50):
        """Train with minimal steps for speed."""
        # Load data
        data_seq = []
        for f in train_files:
            if os.path.exists(f):
                with open(f, 'rb') as file:
                    data_seq.append(pickle.load(file))
        
        if len(data_seq) < 7:
            return None
        
        # Create model
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Quick training loop
        gamma = 0.3
        for step in range(steps):
            model.train()
            optimizer.zero_grad()
            total_loss = 0
            
            for i, data in enumerate(data_seq):
                features = data['trainFeature'].to(device)
                series = data['train_in_portfolio_series'].to(device)
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                ret = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                
                ret_series = torch.sum(series[:, 1:] * shares - series[:, 0:1] * shares, dim=0)
                sharpe = ret / (torch.std(ret_series) + 1e-10)
                
                loss = -sharpe * (gamma ** (len(data_seq) - i - 1))
                total_loss += loss
            
            total_loss.backward()
            optimizer.step()
        
        return model
    
    # Evaluate function
    def evaluate(model, test_file):
        if not os.path.exists(test_file):
            return None
        
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        if test_data.get('test_in_portfolio_series') is None:
            return None
        
        model.eval()
        with torch.no_grad():
            features = test_data['testFeature'].to(device)
            series = test_data['test_in_portfolio_series'].to(device)
            
            weights = model(features)
            shares = weights / (series[:, 0:1] + 1e-10)
            ret = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            
            # Apply transaction cost
            ret = ret - 0.0015
            
            return ret.item()
    
    # Run experiment for each year
    results = {'years': {}}
    
    for year in [2022, 2023, 2024]:
        print(f"\n{'='*60}")
        print(f"Year {year} - Monthly Trading (12x)")
        print(f"{'='*60}")
        
        year_data = {
            'trades': [],
            'cumulative': 1.0
        }
        
        for month in range(1, 13):
            print(f"{year}-{month:02d}: ", end='')
            
            train_files, test_file = find_month_files(year, month)
            
            if not train_files:
                print("No data")
                continue
            
            # Train model (fast)
            model = quick_train(train_files, steps=50)
            
            if model is None:
                print("Training failed")
                continue
            
            # Evaluate
            ret = evaluate(model, test_file)
            
            if ret is not None:
                year_data['cumulative'] *= (1 + ret)
                year_data['trades'].append({
                    'month': month,
                    'return': ret,
                    'cumulative': year_data['cumulative'] - 1
                })
                
                print(f"Return: {ret*100:+.2f}% | YTD: {(year_data['cumulative']-1)*100:+.2f}%")
            else:
                print("Eval failed")
            
            # Clean memory
            del model
            torch.cuda.empty_cache()
        
        # Calculate statistics
        if year_data['trades']:
            returns = [t['return'] for t in year_data['trades']]
            year_data['stats'] = {
                'annual_return': year_data['cumulative'] - 1,
                'num_trades': len(returns),
                'avg_return': np.mean(returns),
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-10) if len(returns) > 1 else 0,
                'win_rate': sum(1 for r in returns if r > 0) / len(returns)
            }
            
            results['years'][year] = year_data
    
    # Overall statistics
    all_trades = []
    overall_cumulative = 1.0
    for year_data in results['years'].values():
        all_trades.extend(year_data['trades'])
        overall_cumulative *= year_data['cumulative']
    
    if all_trades:
        all_returns = [t['return'] for t in all_trades]
        results['overall'] = {
            'total_trades': len(all_trades),
            'cumulative_return': overall_cumulative - 1,
            'avg_return': np.mean(all_returns),
            'sharpe': np.mean(all_returns) / (np.std(all_returns) + 1e-10),
            'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns)
        }
    
    # Save results
    with open(f'{output_dir}results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY - SEQUENTIAL SUPERVISED (Monthly 12x/year)")
    print("="*80)
    
    for year, data in results['years'].items():
        if 'stats' in data:
            stats = data['stats']
            print(f"\n{year}:")
            print(f"  Annual return: {stats['annual_return']*100:+.2f}%")
            print(f"  Trades: {stats['num_trades']}")
            print(f"  Avg: {stats['avg_return']*100:+.2f}%")
            print(f"  Sharpe: {stats['sharpe']:.3f}")
            print(f"  Win rate: {stats['win_rate']*100:.0f}%")
    
    if 'overall' in results:
        overall = results['overall']
        print(f"\nOverall (2022-2024):")
        print(f"  Total trades: {overall['total_trades']}")
        print(f"  Cumulative: {overall['cumulative_return']*100:+.2f}%")
        print(f"  Avg per trade: {overall['avg_return']*100:+.2f}%")
        print(f"  Sharpe: {overall['sharpe']:.3f}")
        print(f"  Win rate: {overall['win_rate']*100:.0f}%")
        
        # Compare with DQN
        print(f"\nDQN/TD3 average: -4.52%")
        print(f"Sequential (this): {overall['avg_return']*100:+.2f}%")
        print(f"Improvement: {(overall['avg_return'] + 0.0452)*100:+.2f} percentage points")
    
    print(f"\nResults saved to: {output_dir}")
    
    return results


if __name__ == "__main__":
    results = run_fast_monthly_experiment()