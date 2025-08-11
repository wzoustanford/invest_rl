"""
LSTM Sequential Model - 2023 and 2024 Trading Years
750 iterations, gamma=0.3 to match Sequential Supervised
CORRECT DATE ALIGNMENT: test_data_start_date is 1 year before actual trading
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
output_dir = f'{data_dir}lstm_2023_2024_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"Configuration: LSTM with gamma={GAMMA}, {TRAINING_STEPS} training steps")
print(f"LSTM: hidden_dim={LSTM_HIDDEN_DIM}, layers={NUM_LSTM_LAYERS}")
print(f"Gradient clipping: max_norm=5.0")
print("\n" + "="*80)
print("TESTING 2023 AND 2024 TRADING YEARS")
print("Date Alignment: test_data_start_date + 1 year = actual trading date")
print("="*80 + "\n")

def get_date_from_filename(filename):
    """Extract test date from filename."""
    if 'test_data_start_date_' in filename:
        date_str = filename.split('test_data_start_date_')[1].split('_news')[0]
        return date_str.replace('_', '-')
    return None

def find_all_available_months(target_trading_year):
    """
    Find ALL available months for a given TRADING year.
    CRITICAL: Files are from 1 year before trading year!
    
    For 2023 trading: use files with test_data_start_date_2022_XX_XX
    For 2024 trading: use files with test_data_start_date_2023_XX_XX
    """
    available_months = {}
    target_file_year = target_trading_year - 1  # FILES ARE 1 YEAR BEFORE TRADING!
    
    print(f"\nSearching for {target_trading_year} trading data...")
    print(f"Looking for files with year {target_file_year} in test_data_start_date")
    
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
                print(f"Error loading {filepath}: {e}")
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
    
    # Find minimum number of stocks across all timesteps
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
        # Get final timestep weights
        weights = model([features], return_all_timesteps=False)
        
        # Calculate returns
        initial_prices = series[:, 0:1] + 1e-10
        shares = weights / initial_prices
        final_prices = series[:, -1:]
        returns = (final_prices - initial_prices) * shares
        total_return = torch.sum(returns).item()
        
        # Apply transaction cost
        total_return = total_return - 0.0015  # 0.15% transaction cost
        
        # Count stocks selected (>1% allocation)
        num_stocks = (weights > 0.01).sum().item()
    
    return {
        'return': total_return * 100,  # Convert to percentage
        'num_stocks': num_stocks
    }

# Process 2023 and 2024
all_results = {}

for year in [2023, 2024]:
    print("=" * 80)
    print(f"Processing {year} Trading Year")
    print(f"Using files with test_data_start_date_{year-1}_XX_XX")
    print("=" * 80)
    
    # Find all available months
    available_months = find_all_available_months(year)
    
    if not available_months:
        print(f"ERROR: No data files found for {year} trading (need {year-1} files)")
        continue
    
    print(f"Found {len(available_months)} months for {year}: {sorted(available_months.keys())}")
    print()
    
    results = {
        'year': year,
        'file_year_used': year - 1,
        'monthly_returns': {},
        'trades': []
    }
    
    # Process each available month
    for month in sorted(available_months.keys()):
        idx, date_str, filepath = available_months[month]
        
        print(f"\n{year}-{month:02d}: Training with 750 iterations")
        print(f"  Using file: {os.path.basename(filepath)}")
        print(f"  File date: {date_str} → Trading date: {year}-{month:02d}")
        
        # Get 7 consecutive files for LSTM training
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
        
        # Evaluate on the last file (which is the test period)
        test_result = evaluate_on_test_data(model, train_files[-1])
        
        if test_result:
            results['monthly_returns'][month] = test_result['return']
            results['trades'].append({
                'month': month,
                'return': test_result['return'],
                'num_stocks': test_result['num_stocks'],
                'file_date': date_str,
                'actual_trading_month': f"{year}-{month:02d}"
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
        
        print(f"\n{year} Summary:")
        print(f"  Annual Return: {annual_return * 100:+.2f}%")
        print(f"  Win Rate: {win_rate * 100:.0f}%")
        print(f"  Best Month: {max(returns_list):+.2f}%")
        print(f"  Worst Month: {min(returns_list):+.2f}%")
    
    all_results[str(year)] = results

# Save results
results_file = f'{output_dir}lstm_2023_2024_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nResults saved to: {results_file}")

# Create comparison report
report_file = f'{output_dir}lstm_2023_2024_report.md'
with open(report_file, 'w') as f:
    f.write("# LSTM Results for 2023-2024 Trading Years\n\n")
    f.write("## Configuration\n")
    f.write("- 750 training iterations, γ=0.3\n")
    f.write("- LSTM: 64 hidden dims, 2 layers, gradient clipping (max_norm=5.0)\n")
    f.write("- Transaction costs: 0.15% per trade\n\n")
    
    f.write("## Important: Date Alignment\n")
    f.write("- 2023 trading uses files with test_data_start_date_2022_XX_XX\n")
    f.write("- 2024 trading uses files with test_data_start_date_2023_XX_XX\n\n")
    
    for year_str, results in all_results.items():
        if 'summary' in results:
            s = results['summary']
            f.write(f"\n## {year_str} Results\n\n")
            f.write(f"- Annual Return: {s['annual_return']:+.2f}%\n")
            f.write(f"- Win Rate: {s['win_rate']*100:.0f}%\n")
            f.write(f"- Best Month: {s['best_month']:+.2f}%\n")
            f.write(f"- Worst Month: {s['worst_month']:+.2f}%\n")
            f.write(f"- Months Traded: {s['num_months']}\n\n")
            
            f.write("### Monthly Details\n\n")
            if 'trades' in results:
                for trade in results['trades']:
                    f.write(f"- {trade['actual_trading_month']}: {trade['return']:+.2f}% ({trade['num_stocks']} stocks)\n")

print(f"Report saved to: {report_file}")
print("\nDone!")