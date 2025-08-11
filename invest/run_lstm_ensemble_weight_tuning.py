"""
LSTM Ensemble with Weight Fine-tuning
1. Train 2L, 3L, 4L models separately (independently)
2. Freeze all model weights
3. Fine-tune only the ensemble weights using the same Sharpe loss
4. Perform inference with frozen models and optimized weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Model configuration
NUM_TIMESTEPS = 7
LSTM_HIDDEN_DIM = 64
DROPOUT_RATIO = 0.0
NUM_CONV_FILTERS = 32
GAMMA = 0.3

# Training configuration
INDIVIDUAL_TRAINING_STEPS = 750  # Steps for each individual model
WEIGHT_TUNING_STEPS = 200  # Steps for weight optimization
LEARNING_RATE = 0.001
WEIGHT_LEARNING_RATE = 0.01  # Higher LR for weight tuning since fewer parameters

# Data configuration
data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
output_dir = f'{data_dir}lstm_ensemble_weight_tuned_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"=" * 80)
print(f"LSTM ENSEMBLE WITH WEIGHT FINE-TUNING")
print(f"Strategy: Train models separately, then optimize weights only")
print(f"Individual training: {INDIVIDUAL_TRAINING_STEPS} steps per model")
print(f"Weight tuning: {WEIGHT_TUNING_STEPS} steps")
print(f"=" * 80)
print(f"Testing years: 2021, 2022, 2023, 2024, 2025")
print(f"=" * 80 + "\n")


class WeightTunedEnsemble(nn.Module):
    """
    Ensemble with separately trained models and learnable weights
    """
    
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.models = {}
        
        # Learnable weights (will be optimized after models are trained)
        # Initialize with equal weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3.0)
        
    def add_model(self, num_layers, model):
        """Add a pre-trained model and freeze its weights"""
        model = model.to(self.device)
        # Freeze all model parameters
        for param in model.parameters():
            param.requires_grad = False
        model.eval()  # Set to eval mode
        self.models[num_layers] = model
        
    def forward(self, features_list):
        """Forward pass with frozen models and learnable weights"""
        predictions = []
        
        # Get predictions from each frozen model
        with torch.no_grad():
            for num_layers in [2, 3, 4]:
                if num_layers in self.models:
                    pred = self.models[num_layers](features_list, return_all_timesteps=False)
                    predictions.append(pred)
        
        # Apply softmax to weights to ensure they sum to 1 and are positive
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Combine predictions with learnable weights
        ensemble_pred = weights[0] * predictions[0]
        for i in range(1, len(predictions)):
            ensemble_pred = ensemble_pred + weights[i] * predictions[i]
        
        # Renormalize to ensure portfolio weights sum to 1
        ensemble_pred = F.softmax(ensemble_pred, dim=-1)
        
        return ensemble_pred
    
    def get_weights(self):
        """Get the current ensemble weights"""
        with torch.no_grad():
            weights = F.softmax(self.ensemble_weights, dim=0)
            return weights.cpu().numpy()


def train_individual_model(num_layers, train_data, device):
    """Train a single LSTM model independently"""
    print(f"  Training {num_layers}-layer model independently...")
    
    model = LSTMSequentialModel(
        num_conv_filters=NUM_CONV_FILTERS,
        hidden_dim=47,
        lstm_hidden_dim=LSTM_HIDDEN_DIM,
        num_lstm_layers=num_layers,
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
    for step in range(INDIVIDUAL_TRAINING_STEPS):
        loss = trainer.train_step(train_data)
        loss_history.append(loss)
        
        if (step + 1) % 250 == 0:
            print(f"    {num_layers}L - Step {step+1}/{INDIVIDUAL_TRAINING_STEPS}, Loss: {loss:.4f}")
    
    final_loss = loss_history[-1]
    print(f"    {num_layers}L - Training complete. Final loss: {final_loss:.4f}")
    
    return model, loss_history, final_loss


def compute_sharpe_loss(weights, price_series, gamma, device):
    """Compute gamma-discounted Sharpe ratio loss"""
    initial_prices = price_series[:, 0:1] + 1e-10
    shares = weights / initial_prices
    
    returns_matrix = (price_series[:, 1:] - price_series[:, :-1]) * shares.unsqueeze(1).expand(-1, price_series.shape[1]-1)
    
    period_returns = torch.sum(returns_matrix, dim=0)
    
    # Gamma discounting
    T = len(period_returns)
    gamma_weights = torch.tensor([gamma ** t for t in range(T)], 
                                dtype=torch.float32, device=device)
    gamma_weights = gamma_weights / gamma_weights.sum()
    
    weighted_mean = torch.sum(period_returns * gamma_weights)
    weighted_variance = torch.sum(((period_returns - weighted_mean) ** 2) * gamma_weights)
    weighted_std = torch.sqrt(weighted_variance + 1e-10)
    
    sharpe_ratio = weighted_mean / weighted_std
    
    return -sharpe_ratio


def fine_tune_ensemble_weights(ensemble, train_data, device):
    """Fine-tune only the ensemble weights with frozen models"""
    print("\n  Fine-tuning ensemble weights (models frozen)...")
    
    # Only optimize the ensemble weights, not the model parameters
    optimizer = torch.optim.Adam([ensemble.ensemble_weights], lr=WEIGHT_LEARNING_RATE)
    
    # Extract features and price series
    features_list = []
    for data in train_data:
        features = data['features'].to(device)
        features_list.append(features)
    
    price_series = train_data[-1]['price_series'].to(device)
    
    loss_history = []
    weight_history = []
    
    for step in range(WEIGHT_TUNING_STEPS):
        optimizer.zero_grad()
        
        # Get ensemble prediction (models are frozen, only weights update)
        ensemble_pred = ensemble(features_list)
        
        # Compute Sharpe loss
        loss = compute_sharpe_loss(ensemble_pred, price_series, GAMMA, device)
        
        # Backward pass (only updates ensemble weights)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([ensemble.ensemble_weights], max_norm=1.0)
        
        optimizer.step()
        
        loss_history.append(loss.item())
        weight_history.append(ensemble.get_weights().copy())
        
        if (step + 1) % 50 == 0 or step == 0:
            weights = ensemble.get_weights()
            print(f"    Step {step+1}/{WEIGHT_TUNING_STEPS}, Loss: {loss.item():.4f}, "
                  f"Weights: [2L: {weights[0]:.3f}, 3L: {weights[1]:.3f}, 4L: {weights[2]:.3f}]")
    
    final_weights = ensemble.get_weights()
    print(f"  Weight tuning complete. Final weights: "
          f"[2L: {final_weights[0]:.3f}, 3L: {final_weights[1]:.3f}, 4L: {final_weights[2]:.3f}]")
    
    return loss_history, weight_history


def evaluate_ensemble_and_individuals(ensemble, test_file_path, device):
    """Evaluate ensemble and individual models on test data"""
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
    
    initial_prices = series[:, 0:1] + 1e-10
    final_prices = series[:, -1:]
    
    results = {}
    
    # Evaluate ensemble
    ensemble.eval()
    with torch.no_grad():
        ensemble_weights = ensemble([features])
        
        shares = ensemble_weights / initial_prices
        returns = (final_prices - initial_prices) * shares
        total_return = torch.sum(returns).item() - 0.0015
        
        num_stocks = (ensemble_weights > 0.01).sum().item()
        top10_weight = torch.topk(ensemble_weights.flatten(), min(10, len(ensemble_weights))).values.sum().item()
        max_weight = ensemble_weights.max().item()
        
        results['ensemble'] = {
            'return': total_return * 100,
            'num_stocks': num_stocks,
            'top10_weight': top10_weight * 100,
            'max_weight': max_weight * 100
        }
    
    # Evaluate individual models
    for num_layers, model in ensemble.models.items():
        model.eval()
        with torch.no_grad():
            individual_weights = model([features], return_all_timesteps=False)
            
            shares = individual_weights / initial_prices
            returns = (final_prices - initial_prices) * shares
            total_return = torch.sum(returns).item() - 0.0015
            
            num_stocks = (individual_weights > 0.01).sum().item()
            
            results[f'{num_layers}L'] = {
                'return': total_return * 100,
                'num_stocks': num_stocks
            }
    
    return results


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


# Main processing loop
all_results = {}

for year in [2021, 2022, 2023, 2024, 2025]:
    print("=" * 80)
    print(f"Processing {year} Trading Year")
    print("=" * 80)
    
    available_months = find_all_available_months(year)
    
    if not available_months:
        print(f"ERROR: No data files found for {year} trading")
        continue
    
    print(f"Found {len(available_months)} months for {year}: {sorted(available_months.keys())}")
    
    results = {
        'year': year,
        'architecture': 'Weight-Tuned-Ensemble',
        'monthly_returns': {},
        'ensemble_trades': [],
        'individual_monthly': {'2L': [], '3L': [], '4L': []},
        'weight_evolution': []
    }
    
    for month in sorted(available_months.keys()):
        idx, date_str, filepath = available_months[month]
        
        # Skip months as before
        if year == 2021 and month < 5:
            print(f"\n{year}-{month:02d}: Skipping (before May)")
            continue
        if year == 2025 and month > 5:
            print(f"\n{year}-{month:02d}: Skipping (future date)")
            continue
        
        print(f"\n{year}-{month:02d}: Processing")
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
        
        # Step 1: Train individual models separately
        print("\n  Step 1: Training individual models...")
        models = {}
        individual_losses = {}
        
        for num_layers in [2, 3, 4]:
            model, loss_hist, final_loss = train_individual_model(num_layers, train_data, device)
            models[num_layers] = model
            individual_losses[num_layers] = final_loss
        
        # Step 2: Create ensemble and add frozen models
        print("\n  Step 2: Creating ensemble with frozen models...")
        ensemble = WeightTunedEnsemble(device=device)
        for num_layers, model in models.items():
            ensemble.add_model(num_layers, model)
        
        # Step 3: Fine-tune ensemble weights only
        print("\n  Step 3: Fine-tuning ensemble weights...")
        weight_loss_history, weight_evolution = fine_tune_ensemble_weights(ensemble, train_data, device)
        
        # Step 4: Evaluate on test data
        print("\n  Step 4: Evaluating on test data...")
        test_results = evaluate_ensemble_and_individuals(ensemble, train_files[-1], device)
        
        if test_results:
            # Store results
            results['monthly_returns'][month] = test_results['ensemble']['return']
            
            final_weights = ensemble.get_weights()
            results['ensemble_trades'].append({
                'month': month,
                'date': f"{year}-{month:02d}",
                'ensemble_return': test_results['ensemble']['return'],
                'num_stocks': test_results['ensemble']['num_stocks'],
                'top10_weight': test_results['ensemble']['top10_weight'],
                'max_weight': test_results['ensemble']['max_weight'],
                '2L_return': test_results['2L']['return'],
                '3L_return': test_results['3L']['return'],
                '4L_return': test_results['4L']['return'],
                'final_weights': {
                    '2L': float(final_weights[0]),
                    '3L': float(final_weights[1]),
                    '4L': float(final_weights[2])
                },
                'weight_tuning_loss': weight_loss_history[-1],
                'individual_training_losses': individual_losses
            })
            
            # Track individual model performance
            for model_name in ['2L', '3L', '4L']:
                results['individual_monthly'][model_name].append(test_results[model_name]['return'])
            
            # Store weight evolution
            results['weight_evolution'].append({
                'month': month,
                'evolution': weight_evolution
            })
            
            # Print results
            print(f"\n  Results for {year}-{month:02d}:")
            print(f"    Ensemble: {test_results['ensemble']['return']:+.2f}% "
                  f"(Stocks: {test_results['ensemble']['num_stocks']}, "
                  f"Top10: {test_results['ensemble']['top10_weight']:.1f}%)")
            print(f"    Individual: 2L: {test_results['2L']['return']:+.2f}%, "
                  f"3L: {test_results['3L']['return']:+.2f}%, "
                  f"4L: {test_results['4L']['return']:+.2f}%")
            print(f"    Final Weights: 2L: {final_weights[0]:.3f}, "
                  f"3L: {final_weights[1]:.3f}, 4L: {final_weights[2]:.3f}")
        else:
            print(f"  Evaluation failed")
        
        # Clean up memory
        del ensemble, models
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate annual summaries
    if results['monthly_returns']:
        returns_list = list(results['monthly_returns'].values())
        annual_return = np.prod([1 + r/100 for r in returns_list]) - 1
        avg_return = np.mean(returns_list)
        win_rate = sum(1 for r in returns_list if r > 0) / len(returns_list)
        
        results['summary'] = {
            'ensemble_annual_return': annual_return * 100,
            'ensemble_avg_monthly': avg_return,
            'ensemble_win_rate': win_rate,
            'ensemble_best_month': max(returns_list),
            'ensemble_worst_month': min(returns_list),
            'ensemble_sharpe': avg_return / (np.std(returns_list) + 1e-10),
            'num_months': len(returns_list)
        }
        
        # Individual model annual returns
        for model_name in ['2L', '3L', '4L']:
            if results['individual_monthly'][model_name]:
                ind_returns = results['individual_monthly'][model_name]
                ind_annual = np.prod([1 + r/100 for r in ind_returns]) - 1
                results['summary'][f'{model_name}_annual_return'] = ind_annual * 100
                results['summary'][f'{model_name}_win_rate'] = sum(1 for r in ind_returns if r > 0) / len(ind_returns)
        
        print(f"\n{year} Annual Summary:")
        print(f"  Ensemble Annual Return: {results['summary']['ensemble_annual_return']:+.2f}%")
        print(f"  Ensemble Win Rate: {results['summary']['ensemble_win_rate']*100:.0f}%")
        print(f"  Ensemble Sharpe: {results['summary']['ensemble_sharpe']:.2f}")
        
        print(f"\n  Individual Model Annual Returns:")
        for model_name in ['2L', '3L', '4L']:
            if f'{model_name}_annual_return' in results['summary']:
                print(f"    {model_name}: {results['summary'][f'{model_name}_annual_return']:+.2f}% "
                      f"(Win Rate: {results['summary'][f'{model_name}_win_rate']*100:.0f}%)")
        
        # Compare ensemble vs best individual
        individual_annuals = [results['summary'].get(f'{m}_annual_return', -999) 
                            for m in ['2L', '3L', '4L']]
        best_individual = max(individual_annuals)
        improvement = results['summary']['ensemble_annual_return'] - best_individual
        print(f"\n  Ensemble vs Best Individual: {improvement:+.2f}pp")
    
    all_results[str(year)] = results

# Save comprehensive results
results_file = f'{output_dir}weight_tuned_ensemble_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nResults saved to: {results_file}")

# Create detailed report
report_file = f'{output_dir}weight_tuned_ensemble_report.md'
with open(report_file, 'w') as f:
    f.write("# Weight-Tuned LSTM Ensemble Results\n\n")
    f.write("## Strategy\n")
    f.write("1. Train 2L, 3L, 4L models independently (750 iterations each)\n")
    f.write("2. Freeze all model parameters\n")
    f.write("3. Fine-tune only ensemble weights (200 iterations)\n")
    f.write("4. Perform inference with frozen models and optimized weights\n\n")
    
    f.write("## Annual Performance Summary\n\n")
    f.write("| Year | Ensemble | 2L Model | 3L Model | 4L Model | Best Individual | Improvement |\n")
    f.write("|------|----------|----------|----------|----------|-----------------|-------------|\n")
    
    for year_str, results in all_results.items():
        if 'summary' in results:
            s = results['summary']
            ensemble_return = s['ensemble_annual_return']
            
            # Individual returns
            returns_2L = s.get('2L_annual_return', 'N/A')
            returns_3L = s.get('3L_annual_return', 'N/A')
            returns_4L = s.get('4L_annual_return', 'N/A')
            
            # Best individual and improvement
            individual_returns = []
            for m in ['2L', '3L', '4L']:
                key = f'{m}_annual_return'
                if key in s:
                    individual_returns.append(s[key])
            
            if individual_returns:
                best_individual = max(individual_returns)
                improvement = ensemble_return - best_individual
                best_model = ['2L', '3L', '4L'][individual_returns.index(best_individual)]
            else:
                best_individual = 'N/A'
                improvement = 'N/A'
                best_model = 'N/A'
            
            f.write(f"| {year_str} | **{ensemble_return:+.2f}%** | ")
            
            if isinstance(returns_2L, float):
                f.write(f"{returns_2L:+.2f}% | ")
            else:
                f.write(f"{returns_2L} | ")
                
            if isinstance(returns_3L, float):
                f.write(f"{returns_3L:+.2f}% | ")
            else:
                f.write(f"{returns_3L} | ")
                
            if isinstance(returns_4L, float):
                f.write(f"{returns_4L:+.2f}% | ")
            else:
                f.write(f"{returns_4L} | ")
            
            if isinstance(best_individual, float):
                f.write(f"{best_model}: {best_individual:+.2f}% | ")
                f.write(f"{improvement:+.2f}pp |\n")
            else:
                f.write(f"{best_individual} | {improvement} |\n")
    
    f.write("\n## Monthly Weight Evolution\n\n")
    f.write("Shows how the optimized weights evolved for each month:\n\n")
    
    for year_str, results in all_results.items():
        if 'ensemble_trades' in results and results['ensemble_trades']:
            f.write(f"\n### {year_str}\n\n")
            f.write("| Month | 2L Weight | 3L Weight | 4L Weight | Ensemble Return |\n")
            f.write("|-------|-----------|-----------|-----------|----------------|\n")
            
            for trade in results['ensemble_trades']:
                weights = trade['final_weights']
                f.write(f"| {trade['date']} | {weights['2L']:.3f} | {weights['3L']:.3f} | "
                       f"{weights['4L']:.3f} | {trade['ensemble_return']:+.2f}% |\n")
    
    f.write("\n## Key Findings\n\n")
    f.write("1. **Weight Optimization**: How weights adapted to different market conditions\n")
    f.write("2. **Ensemble Benefit**: Compare improvement over best individual model\n")
    f.write("3. **Consistency**: Check if ensemble reduces volatility\n")
    f.write("4. **Model Preferences**: Which models get higher weights in different years\n")

print(f"Report saved to: {report_file}")
print("\n" + "="*80)
print("WEIGHT-TUNED ENSEMBLE COMPLETE")
print("Models trained separately, weights optimized on training data")
print("="*80)