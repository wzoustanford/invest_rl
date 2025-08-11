"""
LSTM Ensemble with Pre-trained Models
Loads separately trained 2L, 3L, 4L models and combines only at inference
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

from model.lstm_sequential_model import LSTMSequentialModel

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.manual_seed(42)
np.random.seed(42)

# Model configuration (must match pre-trained models)
NUM_TIMESTEPS = 7
LSTM_HIDDEN_DIM = 64
DROPOUT_RATIO = 0.0
NUM_CONV_FILTERS = 32
GAMMA = 0.3
LEARNING_RATE = 0.001
TRAINING_STEPS = 750

# Ensemble configuration
ENSEMBLE_STRATEGY = 'performance_weighted'  # 'equal', 'performance_weighted', 'market_adaptive'

# Data configuration
data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
output_dir = f'{data_dir}lstm_ensemble_pretrained_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
os.makedirs(output_dir, exist_ok=True)

# Load data files
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(all_files)} data files")
print(f"=" * 80)
print(f"PRE-TRAINED LSTM ENSEMBLE")
print(f"Strategy: {ENSEMBLE_STRATEGY}")
print(f"Models: 2L, 3L, 4L (each trained separately)")
print(f"=" * 80)
print(f"Testing years: 2021, 2022, 2023, 2024, 2025")
print(f"=" * 80 + "\n")


class PreTrainedEnsemble:
    """
    Ensemble that combines pre-trained models only at inference
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.models = {}
        self.weights = {}
        self.performance_history = {2: [], 3: [], 4: []}
        
    def add_model(self, num_layers, model):
        """Add a pre-trained model to ensemble"""
        self.models[num_layers] = model.to(self.device)
        self.models[num_layers].eval()
        
    def train_individual_model(self, num_layers, train_data):
        """Train a single model from scratch"""
        from model.lstm_sequential_model import LSTMSequentialTrainer
        
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
            device=self.device,
            gamma=GAMMA,
            learning_rate=LEARNING_RATE
        )
        
        print(f"  Training {num_layers}L model...")
        loss_history = []
        for step in range(TRAINING_STEPS):
            loss = trainer.train_step(train_data)
            loss_history.append(loss)
            
            if (step + 1) % 250 == 0:
                print(f"    {num_layers}L - Step {step+1}/{TRAINING_STEPS}, Loss: {loss:.4f}")
        
        print(f"    {num_layers}L - Final Loss: {loss:.4f}")
        self.models[num_layers] = model
        return loss_history
    
    def update_weights(self, strategy='equal', performance_scores=None):
        """Update ensemble weights based on strategy"""
        if strategy == 'equal':
            # Equal weights
            for num_layers in [2, 3, 4]:
                self.weights[num_layers] = 1.0 / 3.0
                
        elif strategy == 'performance_weighted' and performance_scores:
            # Weight by recent performance
            total_score = sum(performance_scores.values())
            if total_score > 0:
                for num_layers in [2, 3, 4]:
                    self.weights[num_layers] = performance_scores.get(num_layers, 0) / total_score
            else:
                # Fallback to equal
                for num_layers in [2, 3, 4]:
                    self.weights[num_layers] = 1.0 / 3.0
                    
        elif strategy == 'market_adaptive':
            # Adaptive based on recent volatility or market regime
            # For now, simplified: more layers in bull, fewer in bear
            # This would need market regime detection
            self.weights[2] = 0.2  # Conservative
            self.weights[3] = 0.5  # Balanced
            self.weights[4] = 0.3  # Aggressive
    
    def predict(self, features_list):
        """Get ensemble prediction"""
        predictions = {}
        
        # Get predictions from each model
        for num_layers, model in self.models.items():
            model.eval()
            with torch.no_grad():
                pred = model(features_list, return_all_timesteps=False)
                predictions[num_layers] = pred
        
        # Combine with weights
        ensemble_pred = None
        for num_layers, pred in predictions.items():
            weight = self.weights.get(num_layers, 1.0/3.0)
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred
        
        # Renormalize to sum to 1
        ensemble_pred = torch.softmax(ensemble_pred, dim=-1)
        
        return ensemble_pred, predictions
    
    def evaluate_individual_performance(self, features_list, price_series):
        """Evaluate each model's performance"""
        performances = {}
        
        for num_layers, model in self.models.items():
            model.eval()
            with torch.no_grad():
                weights = model(features_list, return_all_timesteps=False)
                
                # Calculate return
                initial_prices = price_series[:, 0:1] + 1e-10
                shares = weights / initial_prices
                final_prices = price_series[:, -1:]
                returns = (final_prices - initial_prices) * shares
                total_return = torch.sum(returns).item()
                
                performances[num_layers] = total_return
        
        return performances


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


def evaluate_on_test_data(ensemble, test_file_path):
    """Evaluate ensemble on test data."""
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
    
    # Get ensemble and individual predictions
    ensemble_weights, individual_weights = ensemble.predict([features])
    
    # Calculate ensemble return
    initial_prices = series[:, 0:1] + 1e-10
    shares = ensemble_weights / initial_prices
    final_prices = series[:, -1:]
    returns = (final_prices - initial_prices) * shares
    total_return = torch.sum(returns).item() - 0.0015
    
    num_stocks = (ensemble_weights > 0.01).sum().item()
    top10_weight = torch.topk(ensemble_weights.flatten(), min(10, len(ensemble_weights))).values.sum().item()
    max_weight = ensemble_weights.max().item()
    
    # Calculate individual returns
    individual_returns = {}
    for num_layers, weights in individual_weights.items():
        ind_shares = weights / initial_prices
        ind_returns = (final_prices - initial_prices) * ind_shares
        ind_total = torch.sum(ind_returns).item() - 0.0015
        individual_returns[num_layers] = ind_total * 100
    
    return {
        'return': total_return * 100,
        'num_stocks': num_stocks,
        'top10_weight': top10_weight * 100,
        'max_weight': max_weight * 100,
        'individual_returns': individual_returns
    }


# Process all years
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
        'architecture': f'PreTrained-Ensemble-{ENSEMBLE_STRATEGY}',
        'monthly_returns': {},
        'trades': [],
        'individual_monthly': {2: [], 3: [], 4: []}
    }
    
    # Track performance for adaptive weighting
    rolling_performance = {2: [], 3: [], 4: []}
    
    for month in sorted(available_months.keys()):
        idx, date_str, filepath = available_months[month]
        
        # Skip as before
        if year == 2021 and month < 5:
            continue
        if year == 2025 and month > 5:
            continue
        
        print(f"\n{year}-{month:02d}: Training individual models separately")
        
        train_files = all_files[idx-6:idx+1]
        train_data_raw = load_data_sequence(train_files)
        if train_data_raw is None:
            continue
        
        train_data = prepare_training_data(train_data_raw)
        if train_data is None:
            continue
        
        print(f"  Training on {train_data[0]['features'].shape[0]} stocks")
        
        # Create ensemble and train each model independently
        ensemble = PreTrainedEnsemble(device=device)
        
        # Train each model separately
        for num_layers in [2, 3, 4]:
            loss_history = ensemble.train_individual_model(num_layers, train_data)
        
        # Update weights based on strategy
        if ENSEMBLE_STRATEGY == 'performance_weighted' and len(rolling_performance[2]) > 0:
            # Use recent performance
            recent_perf = {
                2: np.mean(rolling_performance[2][-3:]) if rolling_performance[2] else 0,
                3: np.mean(rolling_performance[3][-3:]) if rolling_performance[3] else 0,
                4: np.mean(rolling_performance[4][-3:]) if rolling_performance[4] else 0
            }
            ensemble.update_weights(ENSEMBLE_STRATEGY, recent_perf)
        else:
            ensemble.update_weights(ENSEMBLE_STRATEGY)
        
        # Evaluate
        test_result = evaluate_on_test_data(ensemble, train_files[-1])
        
        if test_result:
            results['monthly_returns'][month] = test_result['return']
            results['trades'].append({
                'month': month,
                'return': test_result['return'],
                'num_stocks': test_result['num_stocks'],
                'individual_returns': test_result['individual_returns'],
                'weights_used': dict(ensemble.weights)
            })
            
            # Update rolling performance
            for num_layers in [2, 3, 4]:
                perf = test_result['individual_returns'][num_layers]
                rolling_performance[num_layers].append(perf)
                results['individual_monthly'][num_layers].append(perf)
            
            print(f"  Ensemble: {test_result['return']:+.2f}% | " + 
                  f"2L: {test_result['individual_returns'][2]:+.2f}% | " +
                  f"3L: {test_result['individual_returns'][3]:+.2f}% | " +
                  f"4L: {test_result['individual_returns'][4]:+.2f}%")
            print(f"  Weights: 2L={ensemble.weights[2]:.3f}, 3L={ensemble.weights[3]:.3f}, 4L={ensemble.weights[4]:.3f}")
        
        del ensemble
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Calculate summaries
    if results['monthly_returns']:
        returns_list = list(results['monthly_returns'].values())
        annual_return = np.prod([1 + r/100 for r in returns_list]) - 1
        
        results['summary'] = {
            'ensemble_annual': annual_return * 100,
            'ensemble_sharpe': np.mean(returns_list) / (np.std(returns_list) + 1e-10),
            'win_rate': sum(1 for r in returns_list if r > 0) / len(returns_list)
        }
        
        # Individual model summaries
        for num_layers in [2, 3, 4]:
            if results['individual_monthly'][num_layers]:
                ind_returns = results['individual_monthly'][num_layers]
                ind_annual = np.prod([1 + r/100 for r in ind_returns]) - 1
                results['summary'][f'{num_layers}L_annual'] = ind_annual * 100
        
        print(f"\n{year} Summary:")
        print(f"  Ensemble: {results['summary']['ensemble_annual']:+.2f}%")
        for num_layers in [2, 3, 4]:
            if f'{num_layers}L_annual' in results['summary']:
                print(f"  {num_layers}L Model: {results['summary'][f'{num_layers}L_annual']:+.2f}%")
    
    all_results[str(year)] = results

# Save results
results_file = f'{output_dir}pretrained_ensemble_results.json'
with open(results_file, 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n\nResults saved to: {results_file}")
print("="*80)
print("PRE-TRAINED ENSEMBLE COMPLETE")
print("Each model trained independently, combined only at inference")
print("="*80)