"""
LSTM Sequential Model - SIMPLIFIED ARCHITECTURE
1 LSTM layer, 32 hidden dimensions, 750 iterations
Testing if simpler architecture improves convergence
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

# SIMPLIFIED ARCHITECTURE PARAMETERS
GAMMA = 0.3
LEARNING_RATE = 0.001
TRAINING_STEPS = 750
NUM_TIMESTEPS = 7
LSTM_HIDDEN_DIM = 32  # REDUCED from 64
NUM_LSTM_LAYERS = 1   # REDUCED from 2
DROPOUT_RATIO = 0.0
NUM_CONV_FILTERS = 32

# Data configuration
data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
output_dir = f'{data_dir}lstm_750iter_simplified_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"=" * 80)
print(f"SIMPLIFIED LSTM ARCHITECTURE TEST")
print(f"Configuration: gamma={GAMMA}, {TRAINING_STEPS} iterations")
print(f"LSTM: hidden_dim={LSTM_HIDDEN_DIM} (reduced), layers={NUM_LSTM_LAYERS} (reduced)")
print(f"Gradient clipping: max_norm=5.0")
print(f"=" * 80)
print(f"Testing years: 2023, 2024 (key comparison years)")
print(f"=" * 80 + "\n")

def get_date_from_filename(filename):
    """Extract test date from filename."""
    if 'test_data_start_date_' in filename:
        date_str = filename.split('test_data_start_date_')[1].split('_news')[0]
        return date_str.replace('_', '-')
    return None

def find_all_available_months(target_trading_year):
    """Find ALL available months for a given TRADING year."""
    available_months = {}
    target_file_year = target_trading_year - 1
    
    print(f"\nSearching for {target_trading_year} trading data...")
    print(f"Looking for files with year {target_file_year} in test_data_start_date")
    
    for i, filepath in enumerate(all_files):
        date_str = get_date_from_filename(filepath)
        if date_str and i >= 6:
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
    
    min_stocks = min(f.shape[0] for f in all_features)
    
    for i in range(len(data_sequence)):
        prepared_data.append({
            'features': all_features[i][:min_stocks],
            'price_series': all_series[i][:min_stocks]
        })
    
    return prepared_data

def train_lstm_model(train_data):
    """Train SIMPLIFIED LSTM model on prepared data."""
    model = LSTMSequentialModel(
        num_conv_filters=NUM_CONV_FILTERS,
        hidden_dim=47,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,  # 32 instead of 64
        num_lstm_layers=NUM_LSTM_LAYERS,   # 1 instead of 2
        dropout_ratio=DROPOUT_RATIO,
        num_timesteps=NUM_TIMESTEPS
    )
    
    trainer = LSTMSequentialTrainer(
        model=model,
        device=device,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE
    )
    
    loss_history = []
    for step in range(TRAINING_STEPS):
        loss = trainer.train_step(train_data)
        loss_history.append(loss)
        
        if (step + 1) % 150 == 0:
            print(f"    Step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
        elif (step + 1) == TRAINING_STEPS:
            print(f"    Step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
    
    return model, trainer, loss_history

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
        
        total_return = total_return - 0.0015
        
        num_stocks = (weights > 0.01).sum().item()
        top10_weight = torch.topk(weights.flatten(), min(10, len(weights))).values.sum().item()
        max_weight = weights.max().item()
    
    return {
        'return': total_return * 100,
        'num_stocks': num_stocks,
        'top10_weight': top10_weight * 100,
        'max_weight': max_weight * 100
    }

# Process 2023 and 2024 (key comparison years)
all_results = {}

for year in [2023, 2024]:
    print("=" * 80)
    print(f"Processing {year} Trading Year")
    print(f"Using files with test_data_start_date_{year-1}_XX_XX")
    print("=" * 80)
    
    available_months = find_all_available_months(year)
    
    if not available_months:
        print(f"ERROR: No data files found for {year} trading")
        continue
    
    print(f"Found {len(available_months)} months for {year}: {sorted(available_months.keys())}")
    print()
    
    results = {
        'year': year,
        'architecture': '1-layer-32dim',
        'iterations': TRAINING_STEPS,
        'file_year_used': year - 1,
        'monthly_returns': {},
        'trades': [],
        'loss_histories': {}
    }
    
    for month in sorted(available_months.keys()):
        idx, date_str, filepath = available_months[month]
        
        print(f"\n{year}-{month:02d}: Training with 750 iterations (simplified architecture)")
        print(f"  Using file: {os.path.basename(filepath)}")
        print(f"  File date: {date_str} → Trading date: {year}-{month:02d}")
        
        train_files = all_files[idx-6:idx+1]
        
        train_data_raw = load_data_sequence(train_files)
        if train_data_raw is None:
            print(f"  Skipping - data loading failed")
            continue
        
        train_data = prepare_training_data(train_data_raw)
        if train_data is None:
            print(f"  Skipping - data preparation failed")
            continue
        
        print(f"  Training on {train_data[0]['features'].shape[0]} stocks")
        
        model, trainer, loss_history = train_lstm_model(train_data)
        
        test_result = evaluate_on_test_data(model, train_files[-1])
        
        if test_result:
            results['monthly_returns'][month] = test_result['return']
            results['trades'].append({
                'month': month,
                'return': test_result['return'],
                'num_stocks': test_result['num_stocks'],
                'top10_weight': test_result['top10_weight'],
                'max_weight': test_result['max_weight'],
                'file_date': date_str,
                'actual_trading_month': f"{year}-{month:02d}"
            })
            results['loss_histories'][month] = {
                'final_loss': loss_history[-1],
                'min_loss': min(loss_history),
                'max_loss': max(loss_history),
                'convergence_iteration': next((i for i, loss in enumerate(loss_history[50:], 50) 
                                              if abs(loss - loss_history[-1]) < 0.1), TRAINING_STEPS)
            }
            print(f"  Result: Return = {test_result['return']:+.2f}%, Stocks = {test_result['num_stocks']}, "
                  f"Top10 = {test_result['top10_weight']:.1f}%, Max = {test_result['max_weight']:.1f}%")
        else:
            print(f"  Evaluation failed")
        
        del model, trainer
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
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
            'months_traded': sorted(results['monthly_returns'].keys()),
            'sharpe_ratio': avg_return / (np.std(returns_list) + 1e-10)
        }
        
        print(f"\n{year} Summary (Simplified Architecture):")
        print(f"  Annual Return: {annual_return * 100:+.2f}%")
        print(f"  Win Rate: {win_rate * 100:.0f}%")
        print(f"  Best Month: {max(returns_list):+.2f}%")
        print(f"  Worst Month: {min(returns_list):+.2f}%")
        print(f"  Sharpe Ratio: {results['summary']['sharpe_ratio']:.2f}")
    
    all_results[str(year)] = results

# Save results
results_file = f'{output_dir}lstm_750iter_simplified_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nResults saved to: {results_file}")

# Create comparison report
report_file = f'{output_dir}lstm_simplified_comparison.md'
with open(report_file, 'w') as f:
    f.write("# LSTM Simplified Architecture Results\n\n")
    f.write("## Architecture Changes\n")
    f.write("- **LSTM Layers**: 2 → 1 (simplified)\n")
    f.write("- **Hidden Dimension**: 64 → 32 (reduced)\n")
    f.write("- **Iterations**: 750 (unchanged)\n")
    f.write("- **Gamma**: 0.3 (unchanged)\n\n")
    
    f.write("## Comparison Table\n\n")
    f.write("| Year | Architecture | Annual Return | Win Rate | Sharpe | vs Sequential |\n")
    f.write("|------|-------------|---------------|----------|--------|---------------|\n")
    
    # Add previous results for comparison
    prev_results = {
        '2023': {'2-layer-64dim': '+5.31%', 'sequential': '+14.00%'},
        '2024': {'2-layer-64dim': '+19.95%', 'sequential': '+406.17%'}
    }
    
    for year_str, results in all_results.items():
        if 'summary' in results:
            s = results['summary']
            prev = prev_results.get(year_str, {})
            f.write(f"| {year_str} | 1-layer-32dim | {s['annual_return']:+.2f}% | "
                   f"{s['win_rate']*100:.0f}% | {s['sharpe_ratio']:.2f} | "
                   f"vs Seq: {prev.get('sequential', 'N/A')} |\n")
            f.write(f"| {year_str} | 2-layer-64dim | {prev.get('2-layer-64dim', 'N/A')} | - | - | - |\n")
    
    f.write("\n## Monthly Details\n\n")
    for year_str, results in all_results.items():
        if 'summary' in results:
            f.write(f"\n### {year_str} Monthly Returns\n\n")
            f.write("| Month | Return | Stocks>1% | Top10 Weight | Max Position | Final Loss |\n")
            f.write("|-------|--------|-----------|--------------|--------------|------------|\n")
            if 'trades' in results:
                for i, trade in enumerate(results['trades']):
                    month = trade['month']
                    loss_info = results['loss_histories'].get(month, {})
                    f.write(f"| {trade['actual_trading_month']} | {trade['return']:+.2f}% | "
                           f"{trade['num_stocks']} | {trade['top10_weight']:.1f}% | "
                           f"{trade['max_weight']:.1f}% | "
                           f"{loss_info.get('final_loss', 0):.2f} |\n")

print(f"Report saved to: {report_file}")
print("\n" + "="*80)
print("SIMPLIFIED ARCHITECTURE TEST COMPLETE")
print("Compare these results with the 2-layer-64dim architecture")
print("="*80)