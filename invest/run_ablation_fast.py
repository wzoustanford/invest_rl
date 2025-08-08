"""
Fast Ablation Study - Test all configurations with reduced iterations for quick results
"""

import torch
import pickle
import numpy as np
import os
import json
import re
import time
from datetime import datetime
from typing import Dict, List

from model.iimodel import IIMODEL

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
np.random.seed(42)

data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/ablation_fast_{timestamp}/'
os.makedirs(output_dir, exist_ok=True)

# Load all files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"Output: {output_dir}")

def get_date(filename):
    pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
    match = re.search(pattern, filename)
    if match:
        return match.group(1).replace('_', '-')
    return None

def find_month_file(year, month):
    for i, filepath in enumerate(all_files):
        date_str = get_date(filepath)
        if date_str:
            file_year = int(date_str[:4])
            file_month = int(date_str[5:7])
            if file_year == year and file_month == month:
                return i, filepath
    return None, None

def train_model(training_files, gamma, iterations, sequence_days):
    """Train with specified configuration."""
    
    # Load data files (up to sequence_days)
    data_seq = []
    for i in range(min(sequence_days, len(training_files))):
        if os.path.exists(training_files[i]):
            with open(training_files[i], 'rb') as f:
                data_seq.append(pickle.load(f))
    
    if len(data_seq) < min(7, sequence_days):  # Need at least 7 files
        return None
    
    # Create model
    model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for step in range(iterations):
        model.train()
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0).to(device)
        
        for i in range(len(data_seq)):
            features = data_seq[i]['trainFeature'].to(device)
            series = data_seq[i]['train_in_portfolio_series'].to(device)
            
            weights = model(features)
            shares = weights / (series[:, 0:1] + 1e-10)
            ret = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
            
            daily_values = torch.sum(series * shares, dim=0)
            daily_returns = daily_values[1:] - daily_values[:-1]
            sharpe = ret / (torch.std(daily_returns) + 1e-10)
            
            # Gamma discounting
            gamma_power = gamma ** (len(data_seq) - i - 1)
            loss = -sharpe * gamma_power
            total_loss = total_loss + loss
        
        total_loss.backward()
        optimizer.step()
    
    return model

def evaluate_model(model, test_file):
    """Evaluate model."""
    if not os.path.exists(test_file) or model is None:
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
        
        # Transaction cost
        ret = ret - 0.0015
        
        daily_values = torch.sum(series * shares, dim=0)
        daily_returns = daily_values[1:] - daily_values[:-1]
        sharpe = ret / (torch.std(daily_returns) + 1e-10)
        
        return ret.item(), sharpe.item()

def run_year_experiment(year, gamma, iterations, sequence_days):
    """Run experiment for one year with given configuration."""
    
    year_data = {
        'year': year,
        'gamma': gamma,
        'iterations': iterations,
        'sequence_days': sequence_days,
        'trades': [],
        'cumulative': 1.0
    }
    
    # Sample months for faster execution (every 2 months)
    test_months = [1, 3, 5, 7, 9, 11] if year != 2021 else [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    for month in test_months:
        file_idx, test_file = find_month_file(year, month)
        
        if file_idx is None or file_idx < sequence_days - 1:
            continue
        
        # Get training files
        start_idx = max(0, file_idx - sequence_days + 1)
        training_files = all_files[start_idx:file_idx + 1]
        
        # Train model (reduced iterations for speed)
        fast_iter = min(iterations, 50) if iterations > 100 else iterations
        model = train_model(training_files, gamma, fast_iter, sequence_days)
        
        # Evaluate
        result = evaluate_model(model, test_file)
        
        if result:
            ret, sharpe = result
            year_data['cumulative'] *= (1 + ret)
            year_data['trades'].append({
                'month': month,
                'return': ret,
                'sharpe': sharpe
            })
        
        # Clean memory
        del model
        torch.cuda.empty_cache()
    
    # Calculate statistics
    if year_data['trades']:
        returns = [t['return'] for t in year_data['trades']]
        year_data['annual_return'] = year_data['cumulative'] - 1
        year_data['avg_return'] = np.mean(returns)
        year_data['num_trades'] = len(returns)
        year_data['win_rate'] = sum(1 for r in returns if r > 0) / len(returns)
    
    return year_data

# Run all experiments
results = {
    'experiments': [],
    'summary': {}
}

print("\n" + "="*80)
print("FAST ABLATION STUDY - Sequential Supervised Learning")
print("="*80)

# Configuration grid
configs = [
    # Baseline
    {'name': 'Baseline', 'gamma': 0.3, 'iter': 20, 'seq': 7},
    
    # More iterations (simulate 750 with 50)
    {'name': '750iter', 'gamma': 0.3, 'iter': 50, 'seq': 7},
    
    # Longer sequence
    {'name': '14days', 'gamma': 0.3, 'iter': 20, 'seq': 14},
    
    # Different gamma values
    {'name': 'Gamma0.1', 'gamma': 0.1, 'iter': 20, 'seq': 7},
    {'name': 'Gamma0.5', 'gamma': 0.5, 'iter': 20, 'seq': 7},
]

years = [2021, 2022, 2023]

for config in configs:
    print(f"\n\nConfiguration: {config['name']}")
    print(f"  Gamma={config['gamma']}, Iterations={config['iter']}, Sequence={config['seq']} days")
    print("-" * 60)
    
    config_results = []
    
    for year in years:
        print(f"  Year {year}: ", end='')
        start = time.time()
        
        year_result = run_year_experiment(
            year,
            config['gamma'],
            config['iter'],
            config['seq']
        )
        
        config_results.append(year_result)
        
        print(f"Return={year_result.get('annual_return', 0)*100:+.2f}% "
              f"(Trades={year_result.get('num_trades', 0)}, "
              f"WinRate={year_result.get('win_rate', 0)*100:.0f}%) "
              f"[{time.time()-start:.1f}s]")
    
    # Store configuration results
    results['experiments'].append({
        'config': config,
        'years': config_results,
        'avg_annual_return': np.mean([y.get('annual_return', 0) for y in config_results])
    })

# Generate summary comparison
print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

baseline_avg = results['experiments'][0]['avg_annual_return']

print(f"\nConfiguration             | 2021     | 2022     | 2023     | Average  | vs Baseline")
print("-" * 85)

for exp in results['experiments']:
    config = exp['config']
    years_data = exp['years']
    avg = exp['avg_annual_return']
    diff = (avg - baseline_avg) * 100
    
    print(f"{config['name']:24} | ", end='')
    for y in years_data:
        print(f"{y.get('annual_return', 0)*100:+7.2f}% | ", end='')
    print(f"{avg*100:+7.2f}% | {diff:+6.2f}pp")

# Best configuration
best_config = max(results['experiments'], key=lambda x: x['avg_annual_return'])
print(f"\nBest Configuration: {best_config['config']['name']}")
print(f"  Average Annual Return: {best_config['avg_annual_return']*100:.2f}%")

# Save results
with open(f'{output_dir}results.json', 'w') as f:
    json.dump(results, f, indent=2, default=float)

print(f"\nResults saved to: {output_dir}")

# Additional analysis for 2021
print("\n" + "="*80)
print("2021 DETAILED ANALYSIS (All Configurations)")
print("="*80)

for exp in results['experiments']:
    config = exp['config']
    year_2021 = next((y for y in exp['years'] if y['year'] == 2021), None)
    
    if year_2021 and year_2021.get('trades'):
        print(f"\n{config['name']}:")
        print(f"  Annual Return: {year_2021['annual_return']*100:+.2f}%")
        print(f"  Number of Trades: {year_2021['num_trades']}")
        print(f"  Win Rate: {year_2021['win_rate']*100:.0f}%")
        print(f"  Monthly Returns: ", end='')
        for t in year_2021['trades'][:6]:
            print(f"M{t['month']}:{t['return']*100:+.1f}% ", end='')
        if len(year_2021['trades']) > 6:
            print("...")

print("\nDone!")