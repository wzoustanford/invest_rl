"""
LSTM Ensemble Model - Combining 2L, 3L, 4L Architectures
Tests different weighting strategies to leverage strengths of each depth
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

from model.lstm_ensemble_model import LSTMEnsemble, LSTMEnsembleTrainer

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# ENSEMBLE CONFIGURATION
GAMMA = 0.3
LEARNING_RATE = 0.001
TRAINING_STEPS = 750
NUM_TIMESTEPS = 7
LSTM_HIDDEN_DIM = 64
ENSEMBLE_LAYERS = [2, 3, 4]  # Combine 2-layer, 3-layer, and 4-layer models
DROPOUT_RATIO = 0.0
NUM_CONV_FILTERS = 32

# Weighting strategies to test
WEIGHTING_STRATEGIES = ['equal', 'learned', 'adaptive', 'performance']
SELECTED_STRATEGY = 'adaptive'  # Can be changed to test different strategies

# Training strategy
TRAINING_STRATEGY = 'joint'  # 'joint', 'sequential', 'independent'

# Data configuration
data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
output_dir = f'{data_dir}lstm_ensemble_{SELECTED_STRATEGY}_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"=" * 80)
print(f"LSTM ENSEMBLE MODEL TEST")
print(f"Architecture: Combining {ENSEMBLE_LAYERS} layer models")
print(f"Weighting Strategy: {SELECTED_STRATEGY}")
print(f"Training Strategy: {TRAINING_STRATEGY}")
print(f"Configuration: gamma={GAMMA}, {TRAINING_STEPS} iterations")
print(f"LSTM: hidden_dim={LSTM_HIDDEN_DIM} for all models")
print(f"Gradient clipping: max_norm=5.0")
print(f"=" * 80)
print(f"Testing years: 2021, 2022, 2023, 2024, 2025")
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

def train_ensemble_model(train_data):
    """Train ENSEMBLE LSTM model on prepared data."""
    model = LSTMEnsemble(
        num_conv_filters=NUM_CONV_FILTERS,
        hidden_dim=47,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        dropout_ratio=DROPOUT_RATIO,
        num_timesteps=NUM_TIMESTEPS,
        ensemble_layers=ENSEMBLE_LAYERS,
        weighting_strategy=SELECTED_STRATEGY,
        device=device
    )
    
    trainer = LSTMEnsembleTrainer(
        ensemble_model=model,
        device=device,
        gamma=GAMMA,
        learning_rate=LEARNING_RATE,
        training_strategy=TRAINING_STRATEGY
    )
    
    loss_history = []
    performance_history = []
    
    for step in range(TRAINING_STEPS):
        loss = trainer.train_step(train_data)
        loss_history.append(loss)
        
        # Periodically evaluate individual models for performance tracking
        if (step + 1) % 100 == 0:
            if SELECTED_STRATEGY == 'performance' or step == TRAINING_STEPS - 1:
                performances = trainer.evaluate_individual_models(train_data)
                performance_history.append(performances)
                
        if (step + 1) % 150 == 0:
            print(f"    Step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
        elif (step + 1) == TRAINING_STEPS:
            print(f"    Step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
            
            # Print final weights if using learned or performance strategy
            if SELECTED_STRATEGY == 'learned':
                weights = torch.softmax(model.weight_params, dim=0).detach().cpu().numpy()
                print(f"    Final learned weights: 2L={weights[0]:.3f}, 3L={weights[1]:.3f}, 4L={weights[2]:.3f}")
            elif SELECTED_STRATEGY == 'performance':
                weights = model.performance_weights.detach().cpu().numpy()
                print(f"    Final performance weights: 2L={weights[0]:.3f}, 3L={weights[1]:.3f}, 4L={weights[2]:.3f}")
    
    return model, trainer, loss_history, performance_history

def evaluate_on_test_data(model, test_file_path, return_individual=False):
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
        if return_individual:
            weights, individual_weights = model([features], return_all_timesteps=False, 
                                                return_individual_predictions=True)
        else:
            weights = model([features], return_all_timesteps=False)
        
        initial_prices = series[:, 0:1] + 1e-10
        shares = weights / initial_prices
        final_prices = series[:, -1:]
        returns = (final_prices - initial_prices) * shares
        total_return = torch.sum(returns).item()
        
        total_return = total_return - 0.0015  # Transaction cost
        
        num_stocks = (weights > 0.01).sum().item()
        top10_weight = torch.topk(weights.flatten(), min(10, len(weights))).values.sum().item()
        max_weight = weights.max().item()
        
        result = {
            'return': total_return * 100,
            'num_stocks': num_stocks,
            'top10_weight': top10_weight * 100,
            'max_weight': max_weight * 100
        }
        
        # Calculate individual model returns if requested
        if return_individual:
            individual_returns = []
            for ind_weights in individual_weights:
                ind_shares = ind_weights / initial_prices
                ind_returns = (final_prices - initial_prices) * ind_shares
                ind_total_return = torch.sum(ind_returns).item() - 0.0015
                individual_returns.append(ind_total_return * 100)
            result['individual_returns'] = individual_returns
    
    return result

# Process all years 2021-2025
all_results = {}

for year in [2021, 2022, 2023, 2024, 2025]:
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
        'architecture': f'Ensemble-{SELECTED_STRATEGY}',
        'ensemble_layers': ENSEMBLE_LAYERS,
        'iterations': TRAINING_STEPS,
        'file_year_used': year - 1,
        'monthly_returns': {},
        'trades': [],
        'loss_histories': {},
        'individual_model_returns': {f'{layers}L': [] for layers in ENSEMBLE_LAYERS}
    }
    
    for month in sorted(available_months.keys()):
        idx, date_str, filepath = available_months[month]
        
        # For 2021, skip months before May (to match previous tests)
        if year == 2021 and month < 5:
            print(f"\n{year}-{month:02d}: Skipping (before May)")
            continue
            
        # For 2025, only do available months
        if year == 2025 and month > 5:
            print(f"\n{year}-{month:02d}: Skipping (future date)")
            continue
        
        print(f"\n{year}-{month:02d}: Training Ensemble ({SELECTED_STRATEGY} weighting)")
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
        
        model, trainer, loss_history, perf_history = train_ensemble_model(train_data)
        
        # Evaluate with individual model tracking
        test_result = evaluate_on_test_data(model, train_files[-1], return_individual=True)
        
        if test_result:
            results['monthly_returns'][month] = test_result['return']
            results['trades'].append({
                'month': month,
                'return': test_result['return'],
                'num_stocks': test_result['num_stocks'],
                'top10_weight': test_result['top10_weight'],
                'max_weight': test_result['max_weight'],
                'file_date': date_str,
                'actual_trading_month': f"{year}-{month:02d}",
                'individual_returns': test_result.get('individual_returns', [])
            })
            results['loss_histories'][month] = {
                'final_loss': loss_history[-1],
                'min_loss': min(loss_history),
                'max_loss': max(loss_history),
                'performance_history': perf_history
            }
            
            # Track individual model performance
            if 'individual_returns' in test_result:
                for i, layers in enumerate(ENSEMBLE_LAYERS):
                    results['individual_model_returns'][f'{layers}L'].append(
                        test_result['individual_returns'][i]
                    )
            
            print(f"  Ensemble Result: Return = {test_result['return']:+.2f}%, Stocks = {test_result['num_stocks']}, "
                  f"Top10 = {test_result['top10_weight']:.1f}%, Max = {test_result['max_weight']:.1f}%")
            
            if 'individual_returns' in test_result:
                ind_returns_str = ", ".join([f"{layers}L: {ret:+.2f}%" 
                                            for layers, ret in zip(ENSEMBLE_LAYERS, test_result['individual_returns'])])
                print(f"  Individual Returns: {ind_returns_str}")
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
        
        # Calculate individual model annual returns
        for layers in ENSEMBLE_LAYERS:
            if results['individual_model_returns'][f'{layers}L']:
                ind_returns = results['individual_model_returns'][f'{layers}L']
                ind_annual = np.prod([1 + r/100 for r in ind_returns]) - 1
                results['summary'][f'{layers}L_annual_return'] = ind_annual * 100
        
        print(f"\n{year} Summary (Ensemble-{SELECTED_STRATEGY}):")
        print(f"  Ensemble Annual Return: {annual_return * 100:+.2f}%")
        print(f"  Win Rate: {win_rate * 100:.0f}%")
        print(f"  Best Month: {max(returns_list):+.2f}%")
        print(f"  Worst Month: {min(returns_list):+.2f}%")
        print(f"  Sharpe Ratio: {results['summary']['sharpe_ratio']:.2f}")
        
        # Print individual model performance
        print(f"\n  Individual Model Annual Returns:")
        for layers in ENSEMBLE_LAYERS:
            if f'{layers}L_annual_return' in results['summary']:
                print(f"    {layers}L Model: {results['summary'][f'{layers}L_annual_return']:+.2f}%")
    
    all_results[str(year)] = results

# Save results
results_file = f'{output_dir}ensemble_{SELECTED_STRATEGY}_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nResults saved to: {results_file}")

# Create comprehensive comparison report
report_file = f'{output_dir}ensemble_{SELECTED_STRATEGY}_report.md'
with open(report_file, 'w') as f:
    f.write(f"# LSTM Ensemble Results - {SELECTED_STRATEGY.upper()} Weighting\n\n")
    f.write("## Ensemble Configuration\n")
    f.write(f"- **Models Combined**: {ENSEMBLE_LAYERS} layer LSTMs\n")
    f.write(f"- **Weighting Strategy**: {SELECTED_STRATEGY}\n")
    f.write(f"- **Training Strategy**: {TRAINING_STRATEGY}\n")
    f.write(f"- **Hidden Dimension**: {LSTM_HIDDEN_DIM} for all models\n")
    f.write(f"- **Training Steps**: {TRAINING_STEPS}\n")
    f.write(f"- **Gamma**: {GAMMA}\n\n")
    
    f.write("## Performance Summary\n\n")
    f.write("| Year | Ensemble | 2L Model | 3L Model | 4L Model | vs Best Individual |\n")
    f.write("|------|----------|----------|----------|----------|-------------------|\n")
    
    for year_str, results in all_results.items():
        if 'summary' in results:
            s = results['summary']
            ensemble_return = s['annual_return']
            
            # Get individual returns
            ind_returns = []
            for layers in ENSEMBLE_LAYERS:
                key = f'{layers}L_annual_return'
                if key in s:
                    ind_returns.append(s[key])
                else:
                    ind_returns.append(None)
            
            # Find best individual
            valid_returns = [r for r in ind_returns if r is not None]
            if valid_returns:
                best_individual = max(valid_returns)
                improvement = ensemble_return - best_individual
            else:
                best_individual = None
                improvement = None
            
            f.write(f"| {year_str} | **{ensemble_return:+.2f}%** | ")
            for i, layers in enumerate(ENSEMBLE_LAYERS):
                if ind_returns[i] is not None:
                    f.write(f"{ind_returns[i]:+.2f}% | ")
                else:
                    f.write("N/A | ")
            
            if improvement is not None:
                f.write(f"{improvement:+.2f}pp |\n")
            else:
                f.write("N/A |\n")
    
    f.write("\n## Key Findings\n\n")
    f.write(f"1. **Weighting Strategy**: {SELECTED_STRATEGY}\n")
    if SELECTED_STRATEGY == 'equal':
        f.write("   - Simple average of all models\n")
    elif SELECTED_STRATEGY == 'learned':
        f.write("   - Weights learned during training\n")
    elif SELECTED_STRATEGY == 'adaptive':
        f.write("   - Weights dynamically adjusted based on input features\n")
    elif SELECTED_STRATEGY == 'performance':
        f.write("   - Weights based on individual model performance\n")
    
    f.write("2. **Ensemble vs Individual Models**: Compare annual returns\n")
    f.write("3. **Risk Reduction**: Check if ensemble reduces volatility\n")
    f.write("4. **Consistency**: Analyze win rates across years\n")

print(f"Report saved to: {report_file}")
print("\n" + "="*80)
print(f"ENSEMBLE MODEL TEST COMPLETE ({SELECTED_STRATEGY} weighting)")
print("Compare ensemble performance with individual 2L, 3L, 4L models")
print("="*80)