"""
Fast LSTM Sequential Model with gamma=0.3 for 2021 and 2022
Reduced training steps for quicker results
"""

import torch
import pickle
import numpy as np
import os
import json
from datetime import datetime
import gc
import warnings
warnings.filterwarnings('ignore')

from model.lstm_sequential_model import LSTMSequentialModel, LSTMSequentialTrainer

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# Parameters - reduced for speed
GAMMA = 0.3
LEARNING_RATE = 0.001
TRAINING_STEPS = 50  # Reduced from 750 for faster testing
NUM_TIMESTEPS = 7
LSTM_HIDDEN_DIM = 64
NUM_LSTM_LAYERS = 2
DROPOUT_RATIO = 0.0
NUM_CONV_FILTERS = 32

# Data configuration
data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
output_dir = f'{data_dir}lstm_gamma03_fast_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"Configuration: LSTM with gamma={GAMMA}, {TRAINING_STEPS} training steps (FAST MODE)")
print(f"LSTM: hidden_dim={LSTM_HIDDEN_DIM}, layers={NUM_LSTM_LAYERS}")
print()

def get_date_from_filename(filename):
    """Extract test date from filename."""
    if 'test_data_start_date_' in filename:
        date_str = filename.split('test_data_start_date_')[1].split('_news')[0]
        return date_str.replace('_', '-')
    return None

def find_files_for_year_month(year, month):
    """Find 7 consecutive files ending at the given year/month."""
    target_files = []
    target_idx = None
    
    for i, filepath in enumerate(all_files):
        date_str = get_date_from_filename(filepath)
        if date_str:
            file_year = int(date_str[:4])
            file_month = int(date_str[5:7])
            
            if file_year == year and file_month == month:
                # Need at least 7 files for training
                if i >= 6:
                    target_files = all_files[i-6:i+1]  # 7 consecutive files
                    target_idx = i
                    return target_files, target_idx
    
    return None, None

def load_data_sequence(file_paths):
    """Load a sequence of data files."""
    data_sequence = []
    
    for filepath in file_paths:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                data_sequence.append(data)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                return None
    
    return data_sequence if len(data_sequence) == len(file_paths) else None

def prepare_training_data(data_sequence):
    """Prepare data for LSTM training."""
    prepared_data = []
    
    # First pass: collect all features to find common stocks
    all_features = []
    all_series = []
    
    for data in data_sequence:
        # Get features and price series
        if 'trainFeature' in data and data['trainFeature'] is not None:
            features = data['trainFeature']
            if torch.is_tensor(features):
                features = features.cpu()
            else:
                features = torch.tensor(features, dtype=torch.float32)
        else:
            return None
            
        if 'train_in_portfolio_series' in data and data['train_in_portfolio_series'] is not None:
            price_series = data['train_in_portfolio_series']
            if torch.is_tensor(price_series):
                price_series = price_series.cpu()
            else:
                price_series = torch.tensor(price_series, dtype=torch.float32)
        else:
            return None
        
        all_features.append(features)
        all_series.append(price_series)
    
    # Find minimum number of stocks across all timesteps
    min_stocks = min(f.shape[0] for f in all_features)
    
    # Use the same stocks (first min_stocks) for all timesteps
    for i in range(len(data_sequence)):
        prepared_data.append({
            'features': all_features[i][:min_stocks],
            'price_series': all_series[i][:min_stocks]
        })
    
    return prepared_data

def train_lstm_model(train_data):
    """Train LSTM model on prepared data."""
    # Initialize model
    model = LSTMSequentialModel(
        num_conv_filters=NUM_CONV_FILTERS,
        hidden_dim=47,  # Matches IIMODEL exactly
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout_ratio=DROPOUT_RATIO,
        num_timesteps=NUM_TIMESTEPS
    )
    
    # Initialize trainer
    trainer = LSTMSequentialTrainer(
        model=model,
        device=device,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE
    )
    
    # Training loop
    for step in range(TRAINING_STEPS):
        loss = trainer.train_step(train_data)
        
        if (step + 1) % 10 == 0:
            print(f"    Step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
    
    return model, trainer

def evaluate_on_test_data(model, test_file_path):
    """Evaluate model on test data."""
    if not os.path.exists(test_file_path):
        return None
    
    try:
        with open(test_file_path, 'rb') as f:
            test_data = pickle.load(f)
    except:
        return None
    
    # Get test features and series
    if 'testFeature' in test_data and test_data['testFeature'] is not None:
        features = test_data['testFeature']
        if torch.is_tensor(features):
            features = features.to(device)
        else:
            features = torch.tensor(features, dtype=torch.float32).to(device)
    else:
        return None
    
    if 'test_in_portfolio_series' in test_data and test_data['test_in_portfolio_series'] is not None:
        series = test_data['test_in_portfolio_series']
        if torch.is_tensor(series):
            series = series.to(device)
        else:
            series = torch.tensor(series, dtype=torch.float32).to(device)
    else:
        return None
    
    # Get model prediction (only need final timestep for test)
    model.eval()
    with torch.no_grad():
        # For test, we just use the current timestep features
        weights = model([features], return_all_timesteps=False)
        
        # Calculate returns
        initial_prices = series[:, 0:1] + 1e-10
        shares = weights / initial_prices
        final_prices = series[:, -1:]
        
        returns = (final_prices - initial_prices) * shares
        total_return = torch.sum(returns).item()
        
        # Apply transaction cost
        total_return = total_return - 0.0015  # 0.15% transaction cost
        
        # Get selected stocks
        top_k = min(10, len(weights))
        top_weights, top_indices = torch.topk(weights.squeeze(), top_k)
        
    return {
        'return': total_return * 100,  # Convert to percentage
        'num_stocks': (weights > 0.01).sum().item(),
        'top_stocks': top_indices.cpu().tolist()
    }

# Run experiments for select months in 2021 and 2022
results = {
    'config': {
        'model': 'LSTM Sequential',
        'gamma': GAMMA,
        'training_steps': TRAINING_STEPS,
        'lstm_hidden_dim': LSTM_HIDDEN_DIM,
        'num_lstm_layers': NUM_LSTM_LAYERS,
        'num_timesteps': NUM_TIMESTEPS
    },
    '2021': {
        'monthly_returns': [],
        'trades': []
    },
    '2022': {
        'monthly_returns': [],
        'trades': []
    }
}

# Process 2021 (sample every 2 months for speed)
print("=" * 60)
print("Processing 2021 (sampling every 2 months)...")
print("=" * 60)

for month in [2, 4, 6, 8, 10, 12]:  # Every 2 months
    print(f"\n2021-{month:02d}:")
    
    # Find training files (7 consecutive files ending at this month)
    train_files, test_idx = find_files_for_year_month(2021, month)
    
    if train_files is None:
        print(f"  Skipping - insufficient data")
        continue
    
    # Load and prepare training data
    print(f"  Loading {len(train_files)} training files...")
    train_data_raw = load_data_sequence(train_files)
    
    if train_data_raw is None:
        print(f"  Skipping - data loading failed")
        continue
    
    train_data = prepare_training_data(train_data_raw)
    
    if train_data is None:
        print(f"  Skipping - data preparation failed")
        continue
    
    print(f"  Found {train_data[0]['features'].shape[0]} common stocks")
    
    # Train model
    print(f"  Training LSTM model...")
    model, trainer = train_lstm_model(train_data)
    
    # Evaluate on test data (the last file in sequence is test)
    test_result = evaluate_on_test_data(model, train_files[-1])
    
    if test_result:
        results['2021']['monthly_returns'].append(test_result['return'])
        results['2021']['trades'].append({
            'month': month,
            'return': test_result['return'],
            'num_stocks': test_result['num_stocks']
        })
        print(f"  Return: {test_result['return']:.2f}%")
    else:
        print(f"  Evaluation failed")
    
    # Clear memory
    del model, trainer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Process 2022 (sample every 2 months for speed)
print("\n" + "=" * 60)
print("Processing 2022 (sampling every 2 months)...")
print("=" * 60)

for month in [2, 4, 6, 8, 10, 12]:  # Every 2 months
    print(f"\n2022-{month:02d}:")
    
    # Find training files
    train_files, test_idx = find_files_for_year_month(2022, month)
    
    if train_files is None:
        print(f"  Skipping - insufficient data")
        continue
    
    # Load and prepare training data
    print(f"  Loading {len(train_files)} training files...")
    train_data_raw = load_data_sequence(train_files)
    
    if train_data_raw is None:
        print(f"  Skipping - data loading failed")
        continue
    
    train_data = prepare_training_data(train_data_raw)
    
    if train_data is None:
        print(f"  Skipping - data preparation failed")
        continue
    
    print(f"  Found {train_data[0]['features'].shape[0]} common stocks")
    
    # Train model
    print(f"  Training LSTM model...")
    model, trainer = train_lstm_model(train_data)
    
    # Evaluate on test data
    test_result = evaluate_on_test_data(model, train_files[-1])
    
    if test_result:
        results['2022']['monthly_returns'].append(test_result['return'])
        results['2022']['trades'].append({
            'month': month,
            'return': test_result['return'],
            'num_stocks': test_result['num_stocks']
        })
        print(f"  Return: {test_result['return']:.2f}%")
    else:
        print(f"  Evaluation failed")
    
    # Clear memory
    del model, trainer
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()

# Calculate summary statistics
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

for year in ['2021', '2022']:
    if results[year]['monthly_returns']:
        returns = results[year]['monthly_returns']
        # For sampled months, estimate annual return differently
        avg_return = np.mean(returns)
        estimated_annual = (1 + avg_return/100) ** 12 - 1  # Compound monthly average
        win_rate = sum(1 for r in returns if r > 0) / len(returns)
        best_month = max(returns)
        worst_month = min(returns)
        
        results[year]['summary'] = {
            'estimated_annual_return': estimated_annual * 100,
            'average_monthly': avg_return,
            'win_rate': win_rate,
            'best_month': best_month,
            'worst_month': worst_month,
            'num_trades': len(returns)
        }
        
        print(f"\n{year} Performance (LSTM γ={GAMMA}, sampled months):")
        print(f"  Estimated Annual Return: {estimated_annual*100:.2f}%")
        print(f"  Average Monthly: {avg_return:.2f}%")
        print(f"  Win Rate: {win_rate*100:.1f}%")
        print(f"  Best Month: {best_month:.2f}%")
        print(f"  Worst Month: {worst_month:.2f}%")
        print(f"  Sampled Trades: {len(returns)}")

# Save results
results_file = f'{output_dir}lstm_gamma03_fast_results.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {results_file}")

# Create comparison table
print("\n" + "=" * 60)
print("COMPARISON WITH SEQUENTIAL SUPERVISED (from report)")
print("=" * 60)

comparison_table = """
# LSTM vs Sequential Supervised Comparison (Fast Mode - Sampled Months)

## Configuration
- LSTM Hidden Dim: 64
- LSTM Layers: 2  
- Training Steps: 50 (reduced for speed)
- Gamma: 0.3
- Timesteps: 7
- Sampled Months: 2, 4, 6, 8, 10, 12 (6 per year)

## Results Comparison

| Year | Model | Gamma | Annual Return | Win Rate | Best Month | Worst Month | Notes |
|------|-------|-------|---------------|----------|------------|-------------|-------|
"""

# Add existing results from the gamma comparison report
comparison_table += "| 2021 | Sequential | 0.3 | -16.22% | 33% | +16.14% | -18.64% | 12 months |\n"

# Add LSTM results for 2021
if '2021' in results and 'summary' in results['2021']:
    s = results['2021']['summary']
    comparison_table += f"| 2021 | **LSTM** | **0.3** | **{s['estimated_annual_return']:.2f}%** | **{s['win_rate']*100:.0f}%** | **{s['best_month']:.2f}%** | **{s['worst_month']:.2f}%** | 6 months sampled |\n"

comparison_table += "| 2022 | Sequential | 0.3 | +14.00% | 42% | +56.98% | -16.19% | 12 months |\n"

# Add LSTM results for 2022
if '2022' in results and 'summary' in results['2022']:
    s = results['2022']['summary']
    comparison_table += f"| 2022 | **LSTM** | **0.3** | **{s['estimated_annual_return']:.2f}%** | **{s['win_rate']*100:.0f}%** | **{s['best_month']:.2f}%** | **{s['worst_month']:.2f}%** | 6 months sampled |\n"

comparison_table += "\n\n## Detailed Monthly Returns\n\n"

for year in ['2021', '2022']:
    if results[year]['trades']:
        comparison_table += f"\n### {year} Sampled Monthly Details (LSTM γ={GAMMA})\n"
        for trade in results[year]['trades']:
            comparison_table += f"- Month {trade['month']:02d}: {trade['return']:.2f}% ({trade['num_stocks']} stocks)\n"

print(comparison_table)

# Save comparison to markdown file
comparison_file = f'{output_dir}lstm_comparison_table.md'
with open(comparison_file, 'w') as f:
    f.write(comparison_table)

print(f"\nComparison table saved to: {comparison_file}")