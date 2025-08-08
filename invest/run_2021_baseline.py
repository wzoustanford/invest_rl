"""
Run 2021 baseline for Sequential Supervised Learning
Same configuration as 2022/2023: gamma=0.3, 100 iterations (reduced to 20 for speed), 7-day sequences
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
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/2021_baseline_{timestamp}/'
os.makedirs(output_dir, exist_ok=True)

# Load all files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")

def get_date(filename):
    pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
    match = re.search(pattern, filename)
    if match:
        return match.group(1).replace('_', '-')
    return None

def find_month_file(year, month):
    """Find file for specific month."""
    for i, filepath in enumerate(all_files):
        date_str = get_date(filepath)
        if date_str:
            file_year = int(date_str[:4])
            file_month = int(date_str[5:7])
            if file_year == year and file_month == month:
                return i, filepath
    return None, None

def train_quick(training_files, gamma=0.3, steps=20):
    """Quick training with same configuration as 2022/2023."""
    
    # Load 7 files
    data_seq = []
    for f in training_files:
        if os.path.exists(f):
            with open(f, 'rb') as file:
                data_seq.append(pickle.load(file))
    
    if len(data_seq) != 7:
        return None
    
    # Create model
    model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training
    for step in range(steps):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0).to(device)
        
        # Process each of 7 files with gamma discounting
        for i in range(7):
            features = data_seq[i]['trainFeature'].to(device)
            series = data_seq[i]['train_in_portfolio_series'].to(device)
            
            # Get portfolio weights
            weights = model(features)
            
            # Calculate return (day 25 - day 1)
            shares = weights / (series[:, 0:1] + 1e-10)
            portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            
            # Calculate Sharpe
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
            
            # Gamma discounting
            gamma_power = gamma ** (7 - i - 1)
            loss = -sharpe * gamma_power
            total_loss = total_loss + loss
        
        total_loss.backward()
        optimizer.step()
    
    return model

def evaluate_quick(model, test_file):
    """Quick evaluation."""
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
        
        # Calculate Sharpe
        daily_values = torch.sum(series * shares, dim=0)
        daily_returns = daily_values[1:] - daily_values[:-1]
        sharpe = ret / (torch.std(daily_returns) + 1e-10)
        
        return ret.item(), sharpe.item()

# Run 2021 experiment
results = {
    'year': 2021,
    'configuration': {
        'algorithm': 'Sequential Supervised',
        'gamma': 0.3,
        'iterations': 20,  # Reduced for speed, same as 2022/2023 runs
        'sequence_days': 7,
        'transaction_cost': 0.0015
    },
    'monthly_trades': [],
    'cumulative': 1.0
}

print("\n" + "="*80)
print("2021 BASELINE - Sequential Supervised Learning")
print("Configuration: gamma=0.3, iterations=20, sequence=7 days")
print("="*80)

trades_executed = 0

# Run all 12 months
for month in range(1, 13):
    # Find file for this month
    file_idx, test_file = find_month_file(2021, month)
    
    if file_idx is None or file_idx < 6:
        print(f"Month {month:02d}: No data or insufficient history")
        continue
    
    # Get 7 training files
    training_files = all_files[file_idx-6:file_idx+1]
    
    test_date = get_date(test_file)
    print(f"Month {month:02d} ({test_date}, file {file_idx}): ", end='')
    
    # Train model
    model = train_quick(training_files, gamma=0.3, steps=20)
    
    if model is None:
        print("Training failed")
        continue
    
    # Evaluate
    result = evaluate_quick(model, test_file)
    
    if result is not None:
        ret, sharpe = result
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
        
        print(f"Ret: {ret*100:+.2f}% | Sharpe: {sharpe:.2f} | YTD: {(results['cumulative']-1)*100:+.2f}%")
    else:
        print("Eval failed")
    
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

# Save results
with open(f'{output_dir}results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

# Print summary
print("\n" + "="*80)
print("2021 SUMMARY")
print("="*80)

if 'statistics' in results:
    stats = results['statistics']
    print(f"Annual Return: {stats['annual_return']*100:+.2f}%")
    print(f"Trades Executed: {stats['trades_executed']}/12")
    print(f"Average per Trade: {stats['avg_return']*100:+.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe']:.3f}")
    print(f"Win Rate: {stats['win_rate']*100:.0f}%")
    print(f"Best Month: {stats['best_month']*100:+.2f}%")
    print(f"Worst Month: {stats['worst_month']*100:+.2f}%")

# Comparison with 2022/2023
print("\n" + "="*80)
print("COMPARISON WITH 2022/2023")
print("="*80)

print(f"\nSequential Supervised Annual Returns:")
print(f"  2021: {results.get('statistics', {}).get('annual_return', 0)*100:+.2f}%")
print(f"  2022: +13.87% (from previous experiments)")
print(f"  2023: +10.88% (from previous experiments)")

avg_3year = (results.get('statistics', {}).get('annual_return', 0) + 0.1387 + 0.1088) / 3
print(f"\n3-Year Average: {avg_3year*100:+.2f}%")

print(f"\nResults saved to: {output_dir}")