"""
Sampled monthly trading - evaluate select months for each year
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime

from model.iimodel import IIMODEL


# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
output_dir = f'{data_dir}seq_sampled_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files\n")

# Sample select months for each year (every 2 months = 6 trades/year)
sample_months = {
    2022: [2, 4, 6, 8, 10, 12],  # 6 trades
    2023: [1, 3, 5, 7, 9, 11],   # 6 trades  
    2024: [1, 3]                 # 2 trades (limited data)
}

def get_date(filename):
    """Extract test date from filename."""
    pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
    match = re.search(pattern, filename)
    if match:
        return match.group(1).replace('_', '-')
    return None

def find_month_files(year, month):
    """Find files for a specific month."""
    for i, filepath in enumerate(all_files):
        date_str = get_date(filepath)
        if date_str:
            file_year = int(date_str[:4])
            file_month = int(date_str[5:7])
            if file_year == year and file_month == month:
                if i >= 6:
                    return all_files[i-6:i+1], all_files[i], i
    return None, None, None

def train_fast(train_files, steps=30):
    """Ultra-fast training."""
    # Load only the necessary data
    data_seq = []
    for f in train_files:
        if os.path.exists(f):
            with open(f, 'rb') as file:
                data_seq.append(pickle.load(file))
    
    if len(data_seq) < 7:
        return None
    
    model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    gamma = 0.3
    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0).to(device)
        
        for i in range(len(data_seq)):
            features = data_seq[i]['trainFeature'].to(device)
            series = data_seq[i]['train_in_portfolio_series'].to(device)
            
            weights = model(features)
            shares = weights / (series[:, 0:1] + 1e-10)
            ret = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            ret_series = torch.sum(series[:, 1:] * shares - series[:, 0:1] * shares, dim=0)
            sharpe = ret / (torch.std(ret_series) + 1e-10)
            
            loss = -sharpe * (gamma ** (len(data_seq) - i - 1))
            total_loss = total_loss + loss
        
        total_loss.backward()
        optimizer.step()
    
    return model

def evaluate_fast(model, test_file):
    """Fast evaluation."""
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
        
        ret_series = torch.sum(series[:, 1:] * shares - series[:, 0:1] * shares, dim=0)
        sharpe = ret / (torch.std(ret_series) + 1e-10)
        
        return ret.item(), sharpe.item()

# Run experiment
results = {'years': {}, 'note': 'Sampled months (6 trades/year for estimation)'}

print("="*80)
print("SAMPLED MONTHLY TRADING - Sequential Supervised")
print("Note: Trading every 2 months (6x/year) for faster execution")
print("="*80)

for year, months in sample_months.items():
    print(f"\n{year} - Trading months: {months}")
    print("-"*40)
    
    year_data = {
        'trades': [],
        'cumulative': 1.0,
        'months_traded': months
    }
    
    for month in months:
        train_files, test_file, file_idx = find_month_files(year, month)
        
        if not train_files:
            print(f"{year}-{month:02d}: No data available")
            continue
        
        # Get test date
        test_date = get_date(test_file)
        print(f"{year}-{month:02d} (file {file_idx}, date {test_date}): ", end='')
        
        # Train model
        model = train_fast(train_files, steps=30)
        
        if model is None:
            print("Training failed")
            continue
        
        # Evaluate
        result = evaluate_fast(model, test_file)
        
        if result is not None:
            ret, sharpe = result
            year_data['cumulative'] *= (1 + ret)
            year_data['trades'].append({
                'month': month,
                'date': test_date,
                'file_idx': file_idx,
                'return': ret,
                'sharpe': sharpe,
                'cumulative': year_data['cumulative'] - 1
            })
            
            print(f"Ret: {ret*100:+.2f}% | Sharpe: {sharpe:.2f} | YTD: {(year_data['cumulative']-1)*100:+.2f}%")
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
            'std_return': np.std(returns),
            'sharpe': np.mean(returns) / (np.std(returns) + 1e-10) if len(returns) > 1 else 0,
            'best': max(returns),
            'worst': min(returns),
            'win_rate': sum(1 for r in returns if r > 0) / len(returns)
        }
        
        # Estimate full year (12 trades) from sampled results
        if len(returns) > 0:
            # Scale up to 12 trades
            scale_factor = 12 / len(returns)
            year_data['estimated_12x'] = {
                'annual_return': (1 + year_data['stats']['avg_return']) ** 12 - 1,
                'note': f'Estimated from {len(returns)} trades'
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
        'std_return': np.std(all_returns),
        'sharpe': np.mean(all_returns) / (np.std(all_returns) + 1e-10),
        'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns)
    }
    
    # Estimate monthly (12x) performance
    avg_monthly = np.mean(all_returns)
    results['estimated_monthly_12x'] = {
        'avg_return_per_trade': avg_monthly,
        'annual_return_estimate': (1 + avg_monthly) ** 12 - 1,
        'note': f'Based on {len(all_trades)} sampled trades'
    }

# Save results
with open(f'{output_dir}results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

# Print summary
print("\n" + "="*80)
print("SUMMARY - Sequential Supervised (Sampled Monthly)")
print("="*80)

for year, data in results['years'].items():
    if 'stats' in data:
        stats = data['stats']
        print(f"\n{year} ({stats['num_trades']} trades sampled):")
        print(f"  Actual return: {stats['annual_return']*100:+.2f}%")
        print(f"  Avg per trade: {stats['avg_return']*100:+.2f}%")
        print(f"  Sharpe: {stats['sharpe']:.3f}")
        print(f"  Win rate: {stats['win_rate']*100:.0f}%")
        print(f"  Best/Worst: {stats['best']*100:+.2f}% / {stats['worst']*100:+.2f}%")
        
        if 'estimated_12x' in data:
            est = data['estimated_12x']
            print(f"  Est. 12x annual: {est['annual_return']*100:+.2f}%")

if 'overall' in results:
    overall = results['overall']
    print(f"\nOverall (2022-2024):")
    print(f"  Total sampled trades: {overall['total_trades']}")
    print(f"  Cumulative (sampled): {overall['cumulative_return']*100:+.2f}%")
    print(f"  Avg per trade: {overall['avg_return']*100:+.2f}%")
    print(f"  Sharpe: {overall['sharpe']:.3f}")
    print(f"  Win rate: {overall['win_rate']*100:.0f}%")

if 'estimated_monthly_12x' in results:
    est = results['estimated_monthly_12x']
    print(f"\nEstimated Monthly Trading (12x/year):")
    print(f"  Avg per trade: {est['avg_return_per_trade']*100:+.2f}%")
    print(f"  Annual return: {est['annual_return_estimate']*100:+.2f}%")

print(f"\n" + "="*80)
print("COMPARISON WITH DQN/TD3")
print("="*80)
print(f"DQN/TD3 average return: -4.52%")
if 'overall' in results:
    print(f"Sequential Supervised (sampled): {overall['avg_return']*100:+.2f}%")
    improvement = overall['avg_return'] + 0.0452
    print(f"Improvement: {improvement*100:+.2f} percentage points")

print(f"\nResults saved to: {output_dir}")

# Also print which exact file indices were used
print(f"\n" + "="*80)
print("FILE INDICES USED")
print("="*80)
for year, data in results['years'].items():
    if 'trades' in data:
        print(f"\n{year}:")
        for trade in data['trades']:
            print(f"  Month {trade['month']:2d}: File index {trade['file_idx']:4d} (date: {trade['date']})")

print("\nDone!")