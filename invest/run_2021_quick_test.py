"""
Quick test for 2021 and ablation parameters
"""

import torch
import pickle
import numpy as np
import os
import json
from model.iimodel import IIMODEL

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)

data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

def quick_train_eval(file_idx, gamma, iters, seq_days):
    """Quick training and evaluation."""
    if file_idx < seq_days:
        return None
        
    # Load data
    data_seq = []
    for i in range(file_idx - seq_days + 1, file_idx + 1):
        if i >= 0 and i < len(all_files):
            with open(all_files[i], 'rb') as f:
                data_seq.append(pickle.load(f))
    
    if len(data_seq) < min(7, seq_days):
        return None
    
    # Train
    model = IIMODEL().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for _ in range(iters):
        model.train()
        opt.zero_grad()
        loss = torch.tensor(0.0).to(device)
        
        for j, d in enumerate(data_seq):
            feat = d['trainFeature'].to(device)
            ser = d['train_in_portfolio_series'].to(device)
            w = model(feat)
            sh = w / (ser[:, 0:1] + 1e-10)
            ret = torch.sum((ser[:, -1:] - ser[:, 0:1]) * sh)
            dv = torch.sum(ser * sh, dim=0)
            dr = dv[1:] - dv[:-1]
            shrp = ret / (torch.std(dr) + 1e-10)
            loss = loss - shrp * (gamma ** (len(data_seq) - j - 1))
        
        loss.backward()
        opt.step()
    
    # Evaluate
    model.eval()
    test_data = data_seq[-1]
    if test_data.get('test_in_portfolio_series') is None:
        return None
        
    with torch.no_grad():
        feat = test_data['testFeature'].to(device)
        ser = test_data['test_in_portfolio_series'].to(device)
        w = model(feat)
        sh = w / (ser[:, 0:1] + 1e-10)
        ret = torch.sum((ser[:, -1:] - ser[:, 0:1]) * sh) - 0.0015
        
    del model
    torch.cuda.empty_cache()
    
    return ret.item()

# Test configurations on select 2021 months
print("="*60)
print("2021 QUICK TEST + ABLATION STUDY")
print("="*60)

# Find 2021 file indices (Jan, Apr, Jul, Oct)
test_indices = []
for month in [1, 4, 7, 10]:
    for i, f in enumerate(all_files):
        if f'2021_{month:02d}' in f or f'2021-{month:02d}' in f:
            if 'test_data_start_date_2021' in f:
                test_indices.append((month, i))
                break

print(f"\n2021 Test months found: {[m for m, i in test_indices]}")

# Test configurations
configs = [
    ('Baseline', 0.3, 20, 7),
    ('750iter', 0.3, 50, 7),  # Simulated
    ('14days', 0.3, 20, 14),
    ('Gamma0.1', 0.1, 20, 7),
    ('Gamma0.5', 0.5, 20, 7),
]

results = {}

for name, gamma, iters, seq in configs:
    print(f"\n{name} (γ={gamma}, iter={iters}, seq={seq}):")
    
    config_returns = []
    for month, idx in test_indices:
        ret = quick_train_eval(idx, gamma, iters, seq)
        if ret is not None:
            config_returns.append(ret)
            print(f"  Month {month}: {ret*100:+.2f}%")
    
    if config_returns:
        avg = np.mean(config_returns)
        annual_est = (1 + avg) ** 12 - 1  # Estimate annual from monthly
        results[name] = {
            'avg_monthly': avg,
            'annual_estimate': annual_est,
            'trades': len(config_returns)
        }
        print(f"  Average: {avg*100:+.2f}% monthly, ~{annual_est*100:+.2f}% annual")

# Summary
print("\n" + "="*60)
print("CONFIGURATION COMPARISON")
print("="*60)

baseline = results.get('Baseline', {}).get('annual_estimate', 0)

print(f"{'Config':<15} | {'Monthly Avg':>12} | {'Annual Est':>12} | {'vs Baseline':>12}")
print("-" * 60)

for name, data in results.items():
    diff = (data['annual_estimate'] - baseline) * 100
    print(f"{name:<15} | {data['avg_monthly']*100:+11.2f}% | {data['annual_estimate']*100:+11.2f}% | {diff:+11.2f}pp")

# Find best
if results:
    best = max(results.items(), key=lambda x: x[1]['annual_estimate'])
    print(f"\nBest Config: {best[0]} with {best[1]['annual_estimate']*100:.2f}% estimated annual return")

print("\nNote: These are quick estimates from 4 months of 2021 data")