"""
LSTM Sequential Model - 750 iterations to match Sequential Supervised
With gradient clipping (max_norm=5.0) for stable training
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

# Parameters - matching Sequential Supervised
GAMMA = 0.3
LEARNING_RATE = 0.001
TRAINING_STEPS = 750  # MATCHING SEQUENTIAL SUPERVISED
NUM_TIMESTEPS = 7
LSTM_HIDDEN_DIM = 64
NUM_LSTM_LAYERS = 2
DROPOUT_RATIO = 0.0
NUM_CONV_FILTERS = 32

# Data configuration
data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
output_dir = f'{data_dir}lstm_750iter_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"Configuration: LSTM with gamma={GAMMA}, {TRAINING_STEPS} training steps")
print(f"LSTM: hidden_dim={LSTM_HIDDEN_DIM}, layers={NUM_LSTM_LAYERS}")
print(f"Gradient clipping: max_norm=5.0")
print("\n" + "="*60)
print("MATCHING SEQUENTIAL SUPERVISED: 750 iterations")
print("Date Alignment: test_data_start_date + 1 year = trading date")
print("="*60 + "\n")

def get_date_from_filename(filename):
    """Extract test date from filename."""
    if 'test_data_start_date_' in filename:
        date_str = filename.split('test_data_start_date_')[1].split('_news')[0]
        return date_str.replace('_', '-')
    return None

def find_all_available_months(target_year):
    """Find ALL available months for a given trading year."""
    available_months = {}
    target_file_year = target_year - 1  # Files are 1 year before trading
    
    for i, filepath in enumerate(all_files):
        date_str = get_date_from_filename(filepath)
        if date_str and i >= 6:  # Need at least 7 files for LSTM
            file_year = int(date_str[:4])
            file_month = int(date_str[5:7])
            
            if file_year == target_file_year:
                if file_month not in available_months:
                    available_months[file_month] = (i, date_str, filepath)
    
    return available_months

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
                return None
    
    return data_sequence if len(data_sequence) == len(file_paths) else None

def prepare_training_data(data_sequence):
    """Prepare data for LSTM training."""
    prepared_data = []
    all_features = []
    all_series = []
    
    for data in data_sequence:
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
    
    min_stocks = min(f.shape[0] for f in all_features)
    
    for i in range(len(data_sequence)):
        prepared_data.append({
            'features': all_features[i][:min_stocks],
            'price_series': all_series[i][:min_stocks]
        })
    
    return prepared_data

def train_lstm_model(train_data):
    """Train LSTM model on prepared data."""
    model = LSTMSequentialModel(
        num_conv_filters=NUM_CONV_FILTERS,
        hidden_dim=47,  # Matches IIMODEL exactly
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout_ratio=DROPOUT_RATIO,
        num_timesteps=NUM_TIMESTEPS
    )
    
    trainer = LSTMSequentialTrainer(
        model=model,
        device=device,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE
    )
    
    # Training with progress tracking
    for step in range(TRAINING_STEPS):
        loss = trainer.train_step(train_data)
        
        # Progress updates
        if (step + 1) % 150 == 0:
            print(f"    Step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
        elif (step + 1) == TRAINING_STEPS:
            print(f"    Final step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
    
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
    
    model.eval()
    with torch.no_grad():
        weights = model([features], return_all_timesteps=False)
        initial_prices = series[:, 0:1] + 1e-10
        shares = weights / initial_prices
        final_prices = series[:, -1:]
        returns = (final_prices - initial_prices) * shares
        total_return = torch.sum(returns).item()
        total_return = total_return - 0.0015  # 0.15% transaction cost
        
    return {
        'return': total_return * 100,  # Convert to percentage
        'num_stocks': (weights > 0.01).sum().item()
    }

# Process both years
all_results = {}

for year in [2021, 2022]:
    print("=" * 60)
    print(f"Processing {year} Trading Year (using {year-1} feature data)")
    print("=" * 60)
    
    # Find all available months
    available_months = find_all_available_months(year)
    print(f"Available months for {year}: {sorted(available_months.keys())}")
    print(f"Total: {len(available_months)} months\n")
    
    results = {
        'year': year,
        'monthly_returns': {},
        'trades': []
    }
    
    # Process each available month
    for month in sorted(available_months.keys()):
        idx, date_str, filepath = available_months[month]
        
        print(f"\n{year}-{month:02d}: Starting training with 750 iterations")
        
        # Get 7 consecutive files
        train_files = all_files[idx-6:idx+1]
        
        # Load and prepare data
        train_data_raw = load_data_sequence(train_files)
        if train_data_raw is None:
            print(f"  Skipping - data loading failed")
            continue
        
        train_data = prepare_training_data(train_data_raw)
        if train_data is None:
            print(f"  Skipping - data preparation failed")
            continue
        
        print(f"  Training on {train_data[0]['features'].shape[0]} stocks")
        
        # Train model with 750 iterations
        model, trainer = train_lstm_model(train_data)
        
        # Evaluate
        test_result = evaluate_on_test_data(model, train_files[-1])
        
        if test_result:
            results['monthly_returns'][month] = test_result['return']
            results['trades'].append({
                'month': month,
                'return': test_result['return'],
                'num_stocks': test_result['num_stocks']
            })
            print(f"  Result: Return = {test_result['return']:+.2f}%, Stocks selected = {test_result['num_stocks']}")
        else:
            print(f"  Evaluation failed")
        
        # Clear memory
        del model, trainer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate summary statistics
    if results['monthly_returns']:
        returns_list = list(results['monthly_returns'].values())
        annual_return = np.prod([1 + r/100 for r in returns_list]) - 1
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list)
        
        results['summary'] = {
            'annual_return': annual_return * 100,
            'average_monthly': avg_return,
            'win_rate': win_rate,
            'best_month': max(returns_list),
            'worst_month': min(returns_list),
            'num_months': len(returns_list),
            'months_traded': sorted(results['monthly_returns'].keys())
        }
    
    all_results[str(year)] = results

# Print comprehensive comparison
print("\n" + "=" * 80)
print("FINAL COMPARISON: LSTM (750 iter) vs Sequential Supervised (750 iter)")
print("=" * 80)

# Create comparison table
comparison_data = []

# Add LSTM results
for year_str, results in all_results.items():
    if 'summary' in results:
        s = results['summary']
        comparison_data.append({
            'Year': year_str,
            'Model': 'LSTM γ=0.3',
            'Iterations': 750,
            'Months Traded': f"{s['num_months']} ({','.join(map(str, s['months_traded']))})",
            'Annual Return': f"{s['annual_return']:+.2f}%",
            'Win Rate': f"{s['win_rate']*100:.0f}%",
            'Best Month': f"{s['best_month']:+.2f}%",
            'Worst Month': f"{s['worst_month']:+.2f}%"
        })

# Add Sequential Supervised results (from gamma comparison report)
sequential_results = [
    {
        'Year': '2021',
        'Model': 'Sequential γ=0.3',
        'Iterations': 750,
        'Months Traded': '12 (all)',
        'Annual Return': '-16.22%',
        'Win Rate': '33%',
        'Best Month': '+16.14%',
        'Worst Month': '-18.64%'
    },
    {
        'Year': '2022',
        'Model': 'Sequential γ=0.3',
        'Iterations': 750,
        'Months Traded': '12 (all)',
        'Annual Return': '+14.00%',
        'Win Rate': '42%',
        'Best Month': '+56.98%',
        'Worst Month': '-16.19%'
    }
]

# Print detailed comparison
print("\n| Year | Model | Iterations | Months Traded | Annual Return | Win Rate | Best Month | Worst Month |")
print("|------|-------|------------|---------------|---------------|----------|------------|-------------|")

for data in comparison_data:
    print(f"| {data['Year']} | {data['Model']} | {data['Iterations']} | {data['Months Traded']} | {data['Annual Return']} | {data['Win Rate']} | {data['Best Month']} | {data['Worst Month']} |")

for data in sequential_results:
    print(f"| {data['Year']} | {data['Model']} | {data['Iterations']} | {data['Months Traded']} | {data['Annual Return']} | {data['Win Rate']} | {data['Best Month']} | {data['Worst Month']} |")

# Print key insights
print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

for year_str, results in all_results.items():
    if 'summary' in results:
        s = results['summary']
        print(f"\n{year_str}:")
        print(f"  LSTM (750 iter): {s['annual_return']:+.2f}% on {s['num_months']} months")
        
        if year_str == '2021':
            print(f"  Sequential (750 iter): -16.22% on 12 months")
            if s['annual_return'] > -16.22:
                print(f"  → LSTM outperforms by {s['annual_return'] + 16.22:.2f} percentage points")
            else:
                print(f"  → Sequential outperforms by {-16.22 - s['annual_return']:.2f} percentage points")
        elif year_str == '2022':
            print(f"  Sequential (750 iter): +14.00% on 12 months")
            if s['annual_return'] > 14.00:
                print(f"  → LSTM outperforms by {s['annual_return'] - 14.00:.2f} percentage points")
            else:
                print(f"  → Sequential outperforms by {14.00 - s['annual_return']:.2f} percentage points")

# Save results
results_file = f'{output_dir}lstm_750iter_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nDetailed results saved to: {results_file}")

# Save markdown report
report_file = f'{output_dir}comparison_report_750iter.md'
with open(report_file, 'w') as f:
    f.write("# LSTM vs Sequential Supervised - Equal Training (750 iterations)\n\n")
    f.write("## Configuration\n")
    f.write("- Both models: 750 training iterations, γ=0.3\n")
    f.write("- LSTM: 64 hidden dims, 2 layers, gradient clipping (max_norm=5.0)\n")
    f.write("- Both use 7 consecutive daily files for training\n")
    f.write("- Transaction costs: 0.15% per trade\n\n")
    
    f.write("## Results Table\n\n")
    f.write("| Year | Model | Iterations | Months Traded | Annual Return | Win Rate | Best Month | Worst Month |\n")
    f.write("|------|-------|------------|---------------|---------------|----------|------------|-------------|\n")
    
    for data in comparison_data:
        f.write(f"| {data['Year']} | {data['Model']} | {data['Iterations']} | {data['Months Traded']} | {data['Annual Return']} | {data['Win Rate']} | {data['Best Month']} | {data['Worst Month']} |\n")
    
    for data in sequential_results:
        f.write(f"| {data['Year']} | {data['Model']} | {data['Iterations']} | {data['Months Traded']} | {data['Annual Return']} | {data['Win Rate']} | {data['Best Month']} | {data['Worst Month']} |\n")
    
    f.write("\n## Monthly Details\n\n")
    
    for year_str, results in all_results.items():
        if 'trades' in results and results['trades']:
            f.write(f"### LSTM {year_str} Monthly Returns\n\n")
            for trade in results['trades']:
                f.write(f"- Month {trade['month']:02d}: {trade['return']:+.2f}% ({trade['num_stocks']} stocks selected)\n")
            f.write("\n")

print(f"Comparison report saved to: {report_file}")