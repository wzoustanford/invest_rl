"""
Full Dataset Comparison: LSTM vs Sequential Supervised (gamma=0.3)
Runs both models on the entire dataset from first to last file
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')
import gc

from sequential_supervised_trainer import SequentialSupervisedTrainer, TrainingConfig, OptimizedIIModel
from model.lstm_adaptive_model import LSTMAdaptiveModel


class FullDatasetComparison:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data paths
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/full_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all data files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files", flush=True)
        
        # Extract date range from filenames
        self.start_date = self._extract_date_from_filename(self.all_files[0])
        self.end_date = self._extract_date_from_filename(self.all_files[-1])
        print(f"Date range: {self.start_date} to {self.end_date}", flush=True)
        
        # Model configurations
        self.sequence_length = 7  # Use 7 consecutive days for sequential model
        self.holding_period = 25  # 25-day holding period
        self.feature_dim = 240  # Fixed feature dimension for consistency
        self.num_stocks = 100  # Use top 100 stocks for consistency
        
        # Training configuration
        self.train_ratio = 0.3  # Use first 30% for training
        self.val_ratio = 0.1   # Next 10% for validation
        # Remaining 60% for testing
        
        # Results storage
        self.results = {
            'sequential_supervised': {
                'gamma': 0.3,
                'monthly_returns': [],
                'cumulative_returns': [],
                'trades': []
            },
            'lstm_adaptive': {
                'gamma': 'dynamic',
                'monthly_returns': [],
                'cumulative_returns': [],
                'trades': [],
                'gamma_values': []
            },
            'metadata': {
                'total_files': len(self.all_files),
                'start_date': self.start_date,
                'end_date': self.end_date,
                'train_files': 0,
                'val_files': 0,
                'test_files': 0
            }
        }
    
    def _extract_date_from_filename(self, filename):
        """Extract test date from filename."""
        if 'test_data_start_date_' in filename:
            date_str = filename.split('test_data_start_date_')[1].split('_news')[0]
            return date_str.replace('_', '-')
        return 'unknown'
    
    def load_and_normalize_data(self, file_path):
        """Load and normalize data from a pickle file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Process features and returns
            result = {}
            
            # Try to get features (train or test)
            if 'trainFeature' in data and data['trainFeature'] is not None:
                features = data['trainFeature'].cpu().numpy() if torch.is_tensor(data['trainFeature']) else data['trainFeature']
                result['features'] = self._normalize_features(features)
                
                if 'train_in_portfolio_series' in data and data['train_in_portfolio_series'] is not None:
                    series = data['train_in_portfolio_series'].cpu().numpy() if torch.is_tensor(data['train_in_portfolio_series']) else data['train_in_portfolio_series']
                    result['returns'] = self._calculate_returns(series)
            
            # Also get test data if available
            if 'testFeature' in data and data['testFeature'] is not None:
                features = data['testFeature'].cpu().numpy() if torch.is_tensor(data['testFeature']) else data['testFeature']
                result['test_features'] = self._normalize_features(features)
                
                if 'test_in_portfolio_series' in data and data['test_in_portfolio_series'] is not None:
                    series = data['test_in_portfolio_series'].cpu().numpy() if torch.is_tensor(data['test_in_portfolio_series']) else data['test_in_portfolio_series']
                    result['test_returns'] = self._calculate_returns(series)
            
            return result
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def _normalize_features(self, features):
        """Normalize features to consistent dimensions."""
        # Ensure consistent number of stocks
        if features.shape[0] > self.num_stocks:
            features = features[:self.num_stocks, :]
        elif features.shape[0] < self.num_stocks:
            # Pad with zeros if needed
            padding = np.zeros((self.num_stocks - features.shape[0], features.shape[1]))
            features = np.vstack([features, padding])
        
        # Ensure consistent feature dimension
        if features.shape[1] < self.feature_dim:
            padding = np.zeros((features.shape[0], self.feature_dim - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > self.feature_dim:
            features = features[:, :self.feature_dim]
        
        return features
    
    def _calculate_returns(self, series):
        """Calculate returns from price series."""
        if len(series.shape) == 2 and series.shape[1] > 0:
            returns = (series[:, -1] - series[:, 0]) / (series[:, 0] + 1e-10)
            # Clip extreme returns
            returns = np.clip(returns, -1.0, 10.0)
            
            # Ensure consistent number of stocks
            if len(returns) > self.num_stocks:
                returns = returns[:self.num_stocks]
            elif len(returns) < self.num_stocks:
                padding = np.zeros(self.num_stocks - len(returns))
                returns = np.concatenate([returns, padding])
            
            return returns
        return np.zeros(self.num_stocks)
    
    def train_sequential_supervised(self, train_indices, val_indices, gamma=0.3):
        """Train Sequential Supervised model with gamma=0.3."""
        print(f"\n{'='*60}")
        print(f"Training Sequential Supervised Model (gamma={gamma})")
        print(f"{'='*60}")
        
        # Create model with proper configuration
        config = TrainingConfig(
            gamma=gamma,
            learning_rate=0.001,
            num_steps=750,
            num_consecutive_days=self.sequence_length,
            device=str(self.device),
            dropout_ratio=0.1,
            num_conv_filters=32,
            hidden_dim=47
        )
        
        # Initialize model directly
        model = OptimizedIIModel(config).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        
        # Prepare sequences for training
        print(f"Preparing training sequences from {len(train_indices)} files...", flush=True)
        train_sequences = []
        train_returns = []
        
        total_sequences = (len(train_indices) - self.sequence_length) // 2
        for i in range(0, len(train_indices) - self.sequence_length, 2):  # Step by 2 for efficiency
            if i % 20 == 0:
                print(f"    Processing sequence {i//2}/{total_sequences}...", flush=True)
            sequence_valid = True
            seq_returns = []
            
            # Load sequence of files
            for j in range(self.sequence_length):
                data = self.load_and_normalize_data(self.all_files[train_indices[i+j]])
                if data is None or 'returns' not in data:
                    sequence_valid = False
                    break
                seq_returns.append(data['returns'])
            
            if sequence_valid:
                # Load target file
                target_data = self.load_and_normalize_data(self.all_files[train_indices[i+self.sequence_length]])
                if target_data and 'features' in target_data:
                    train_sequences.append(target_data['features'])
                    train_returns.append(seq_returns)
        
        print(f"Created {len(train_sequences)} training sequences", flush=True)
        
        if len(train_sequences) == 0:
            print("No valid training sequences found")
            return None
        
        # Training loop
        best_val_loss = float('inf')
        for step in range(config.num_steps):
            model.train()
            
            # Random batch selection
            batch_idx = np.random.randint(0, len(train_sequences))
            features = torch.FloatTensor(train_sequences[batch_idx]).to(self.device)
            returns_seq = [torch.FloatTensor(r).to(self.device) for r in train_returns[batch_idx]]
            
            optimizer.zero_grad()
            
            # Calculate cumulative discounted return
            portfolio_weights = model(features)
            cumulative_return = 0
            gamma_power = 1.0
            
            for day_returns in returns_seq:
                portfolio_return = torch.sum(portfolio_weights * day_returns)
                cumulative_return += gamma_power * portfolio_return
                gamma_power *= gamma
            
            # Loss is negative return (maximize return)
            loss = -cumulative_return
            
            # Add regularization
            l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
            loss = loss + 1e-5 * l2_reg
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{config.num_steps}, Loss: {loss.item():.4f}", flush=True)
            
            # Validation every 250 steps
            if (step + 1) % 250 == 0 and len(val_indices) > self.sequence_length:
                val_loss = self._validate_sequential_model(model, val_indices, gamma)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{self.output_dir}/best_sequential_model.pth')
                print(f"    Validation Loss: {val_loss:.4f}", flush=True)
        
        # Load best model if saved
        if os.path.exists(f'{self.output_dir}/best_sequential_model.pth'):
            model.load_state_dict(torch.load(f'{self.output_dir}/best_sequential_model.pth'))
        
        return model
    
    def _validate_sequential_model(self, model, val_indices, gamma):
        """Validate sequential model."""
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i in range(0, min(20, len(val_indices) - self.sequence_length), 2):
                data = self.load_and_normalize_data(self.all_files[val_indices[i]])
                if data and 'features' in data and 'returns' in data:
                    features = torch.FloatTensor(data['features']).to(self.device)
                    returns = torch.FloatTensor(data['returns']).to(self.device)
                    
                    weights = model(features)
                    portfolio_return = torch.sum(weights * returns)
                    loss = -portfolio_return
                    val_losses.append(loss.item())
        
        return np.mean(val_losses) if val_losses else float('inf')
    
    def train_lstm_adaptive(self, train_indices, val_indices):
        """Train LSTM Adaptive model."""
        print(f"\n{'='*60}")
        print(f"Training LSTM Adaptive Model")
        print(f"{'='*60}")
        
        # Prepare sequences
        print(f"Preparing LSTM sequences from {len(train_indices)} files...", flush=True)
        sequences = []
        targets = []
        
        total_sequences = (len(train_indices) - self.sequence_length) // 2
        for i in range(0, len(train_indices) - self.sequence_length, 2):  # Step by 2 for efficiency
            if i % 20 == 0:
                print(f"    Processing LSTM sequence {i//2}/{total_sequences}...", flush=True)
            seq_features = []
            valid = True
            
            # Get sequence of features
            for j in range(self.sequence_length):
                data = self.load_and_normalize_data(self.all_files[train_indices[i+j]])
                if data is None or 'features' not in data:
                    valid = False
                    break
                
                # Average features across stocks for sequence
                avg_features = np.mean(data['features'], axis=0)
                seq_features.append(avg_features)
            
            if valid:
                # Get target returns
                target_data = self.load_and_normalize_data(self.all_files[train_indices[i + self.sequence_length]])
                if target_data and 'returns' in target_data:
                    sequences.append(np.array(seq_features))
                    targets.append(target_data['returns'])
        
        print(f"Created {len(sequences)} LSTM sequences")
        
        if len(sequences) == 0:
            print("No valid LSTM sequences found")
            return None
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Create LSTM model
        model = LSTMAdaptiveModel(
            input_dim=self.feature_dim,
            lstm_hidden_dim=128,
            lstm_layers=2,
            dropout_rate=0.2,
            num_stocks=self.num_stocks,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        num_epochs = 100
        batch_size = 32
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []
            
            # Shuffle indices
            indices = np.random.permutation(len(sequences))
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_seq = torch.FloatTensor(sequences[batch_idx]).to(self.device)
                batch_targets = torch.FloatTensor(targets[batch_idx]).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_seq, return_all_heads=True)
                portfolio_weights = outputs['portfolio_weights']
                portfolio_returns = torch.sum(portfolio_weights * batch_targets, dim=-1)
                
                # Sharpe ratio loss
                returns_mean = portfolio_returns.mean()
                returns_std = portfolio_returns.std() + 1e-8
                sharpe_loss = -returns_mean / returns_std
                
                # Gamma prediction regularization
                if 'gamma_value' in outputs:
                    gamma_reg = torch.mean((outputs['gamma_value'] - 0.3) ** 2)  # Encourage gamma near 0.3
                    total_loss = sharpe_loss + 0.01 * gamma_reg
                else:
                    total_loss = sharpe_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            avg_loss = np.mean(epoch_losses)
            
            # Validation
            if len(val_indices) > self.sequence_length:
                val_loss = self._validate_lstm_model(model, val_indices)
                scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), f'{self.output_dir}/best_lstm_model.pth')
                
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}", flush=True)
            elif (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}", flush=True)
        
        # Load best model if saved
        if os.path.exists(f'{self.output_dir}/best_lstm_model.pth'):
            model.load_state_dict(torch.load(f'{self.output_dir}/best_lstm_model.pth'))
        
        return model
    
    def _validate_lstm_model(self, model, val_indices):
        """Validate LSTM model."""
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for i in range(self.sequence_length, min(self.sequence_length + 20, len(val_indices))):
                # Prepare sequence
                seq_features = []
                valid = True
                
                for j in range(self.sequence_length):
                    data = self.load_and_normalize_data(self.all_files[val_indices[i-self.sequence_length+j]])
                    if data and 'features' in data:
                        avg_features = np.mean(data['features'], axis=0)
                        seq_features.append(avg_features)
                    else:
                        valid = False
                        break
                
                if valid:
                    target_data = self.load_and_normalize_data(self.all_files[val_indices[i]])
                    if target_data and 'returns' in target_data:
                        seq_tensor = torch.FloatTensor([seq_features]).to(self.device)
                        target_returns = torch.FloatTensor([target_data['returns']]).to(self.device)
                        
                        outputs = model(seq_tensor, return_all_heads=True)
                        portfolio_weights = outputs['portfolio_weights']
                        portfolio_return = torch.sum(portfolio_weights * target_returns, dim=-1)
                        
                        loss = -portfolio_return.mean()
                        val_losses.append(loss.item())
        
        return np.mean(val_losses) if val_losses else float('inf')
    
    def backtest_monthly(self, seq_model, lstm_model, test_indices):
        """Backtest both models monthly."""
        print(f"\n{'='*60}")
        print(f"BACKTESTING ({len(test_indices)} test files)")
        print(f"{'='*60}")
        
        # Initialize tracking
        seq_cumulative = 1.0
        lstm_cumulative = 1.0
        
        month_size = 20  # Approximately 20 trading days per month
        num_months = len(test_indices) // month_size
        
        print(f"\nProcessing {num_months} months of data...")
        print(f"{'Month':<8} | {'Sequential (γ=0.3)':>20} | {'LSTM (dynamic γ)':>20}")
        print("-" * 60)
        
        for month in range(num_months):
            month_start = month * month_size
            month_end = min(month_start + month_size, len(test_indices))
            month_indices = test_indices[month_start:month_end]
            
            # Sequential model returns
            seq_returns = []
            lstm_returns = []
            lstm_gammas = []
            
            for idx in month_indices:
                # Sequential model prediction
                data = self.load_and_normalize_data(self.all_files[idx])
                if data and 'test_returns' in data and 'test_features' in data:
                    # Sequential prediction
                    with torch.no_grad():
                        features = torch.FloatTensor(data['test_features']).to(self.device)
                        seq_weights = seq_model(features).cpu().numpy()
                        
                        # Ensure weights sum to 1 and are non-negative
                        seq_weights = np.abs(seq_weights)
                        seq_weights = seq_weights / (np.sum(seq_weights) + 1e-10)
                        
                        # Clip extreme weights to prevent instability
                        seq_weights = np.clip(seq_weights, 0, 0.2)  # Max 20% per stock
                        seq_weights = seq_weights / (np.sum(seq_weights) + 1e-10)
                        
                        seq_return = np.sum(seq_weights * data['test_returns'])
                        # Clip extreme returns
                        seq_return = np.clip(seq_return, -0.5, 0.5)
                        seq_returns.append(seq_return)
                    
                    # LSTM prediction (needs sequence)
                    if idx >= self.sequence_length:
                        seq_features = []
                        valid = True
                        
                        for j in range(self.sequence_length):
                            seq_data = self.load_and_normalize_data(self.all_files[idx - self.sequence_length + j])
                            if seq_data and 'test_features' in seq_data:
                                avg_features = np.mean(seq_data['test_features'], axis=0)
                                seq_features.append(avg_features)
                            else:
                                valid = False
                                break
                        
                        if valid:
                            with torch.no_grad():
                                seq_tensor = torch.FloatTensor([seq_features]).to(self.device)
                                outputs = lstm_model(seq_tensor, return_all_heads=True)
                                lstm_weights = outputs['portfolio_weights'].cpu().numpy()[0]
                                gamma_value = outputs['gamma_value'].cpu().item()
                                
                                lstm_return = np.sum(lstm_weights * data['test_returns'])
                                lstm_returns.append(lstm_return)
                                lstm_gammas.append(gamma_value)
            
            # Calculate monthly averages
            if seq_returns:
                seq_monthly = np.mean(seq_returns) - 0.0015  # Transaction cost
                seq_cumulative *= (1 + seq_monthly)
            else:
                seq_monthly = 0
            
            if lstm_returns:
                lstm_monthly = np.mean(lstm_returns) - 0.0015  # Transaction cost
                lstm_cumulative *= (1 + lstm_monthly)
                avg_gamma = np.mean(lstm_gammas)
            else:
                lstm_monthly = 0
                avg_gamma = 0
            
            # Store results
            self.results['sequential_supervised']['monthly_returns'].append(seq_monthly)
            self.results['sequential_supervised']['cumulative_returns'].append(seq_cumulative - 1)
            self.results['lstm_adaptive']['monthly_returns'].append(lstm_monthly)
            self.results['lstm_adaptive']['cumulative_returns'].append(lstm_cumulative - 1)
            if lstm_gammas:
                self.results['lstm_adaptive']['gamma_values'].extend(lstm_gammas)
            
            # Print monthly results
            print(f"Month {month+1:<2} | {seq_monthly*100:>8.2f}% ({(seq_cumulative-1)*100:>8.2f}%) | "
                  f"{lstm_monthly*100:>8.2f}% ({(lstm_cumulative-1)*100:>8.2f}%)", flush=True)
        
        return seq_cumulative - 1, lstm_cumulative - 1
    
    def calculate_statistics(self):
        """Calculate performance statistics."""
        for model_name in ['sequential_supervised', 'lstm_adaptive']:
            returns = self.results[model_name]['monthly_returns']
            if returns:
                # Filter out extreme outliers for statistics
                filtered_returns = [r for r in returns if abs(r) < 1.0]  # Remove > 100% moves
                
                self.results[model_name]['statistics'] = {
                    'total_return': self.results[model_name]['cumulative_returns'][-1] if self.results[model_name]['cumulative_returns'] else 0,
                    'num_months': len(returns),
                    'avg_monthly_return': np.mean(filtered_returns) if filtered_returns else 0,
                    'std_monthly_return': np.std(filtered_returns) if filtered_returns else 0,
                    'sharpe_ratio': np.mean(filtered_returns) / (np.std(filtered_returns) + 1e-8) * np.sqrt(12) if filtered_returns else 0,
                    'max_monthly_return': max(returns) if returns else 0,
                    'min_monthly_return': min(returns) if returns else 0,
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0,
                    'max_drawdown': self._calculate_max_drawdown(self.results[model_name]['cumulative_returns'])
                }
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """Calculate maximum drawdown."""
        if not cumulative_returns:
            return 0
        
        peak = cumulative_returns[0]
        max_dd = 0
        
        for ret in cumulative_returns:
            if ret > peak:
                peak = ret
            drawdown = (peak - ret) / (1 + peak) if peak > -1 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def run_full_comparison(self):
        """Run the complete comparison on full dataset."""
        print("\n" + "="*80)
        print("FULL DATASET COMPARISON")
        print("LSTM vs Sequential Supervised (γ=0.3)")
        print("="*80)
        
        # Split data
        n_files = len(self.all_files)
        train_end = int(self.train_ratio * n_files)
        val_end = train_end + int(self.val_ratio * n_files)
        
        train_indices = list(range(train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, n_files))
        
        # Update metadata
        self.results['metadata']['train_files'] = len(train_indices)
        self.results['metadata']['val_files'] = len(val_indices)
        self.results['metadata']['test_files'] = len(test_indices)
        
        print(f"\nDataset split:")
        print(f"  Training: {len(train_indices)} files (0-{train_end})")
        print(f"  Validation: {len(val_indices)} files ({train_end}-{val_end})")
        print(f"  Testing: {len(test_indices)} files ({val_end}-{n_files})")
        
        # Train models
        seq_model = self.train_sequential_supervised(train_indices, val_indices, gamma=0.3)
        lstm_model = self.train_lstm_adaptive(train_indices, val_indices)
        
        if seq_model is None or lstm_model is None:
            print("Failed to train one or both models")
            return None
        
        # Backtest
        seq_total, lstm_total = self.backtest_monthly(seq_model, lstm_model, test_indices)
        
        # Calculate statistics
        self.calculate_statistics()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "="*80)
        print("FINAL COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\n{'Metric':<25} | {'Sequential (γ=0.3)':>20} | {'LSTM (dynamic γ)':>20}")
        print("-" * 70)
        
        metrics = [
            ('Total Return (%)', 'total_return', 100),
            ('Num Months', 'num_months', 1),
            ('Avg Monthly Return (%)', 'avg_monthly_return', 100),
            ('Std Monthly Return (%)', 'std_monthly_return', 100),
            ('Sharpe Ratio', 'sharpe_ratio', 1),
            ('Win Rate (%)', 'win_rate', 100),
            ('Max Drawdown (%)', 'max_drawdown', 100),
            ('Best Month (%)', 'max_monthly_return', 100),
            ('Worst Month (%)', 'min_monthly_return', 100)
        ]
        
        for label, key, multiplier in metrics:
            seq_val = self.results['sequential_supervised']['statistics'].get(key, 0) * multiplier
            lstm_val = self.results['lstm_adaptive']['statistics'].get(key, 0) * multiplier
            
            if key == 'num_months':
                print(f"{label:<25} | {seq_val:>20.0f} | {lstm_val:>20.0f}")
            else:
                print(f"{label:<25} | {seq_val:>20.2f} | {lstm_val:>20.2f}")
        
        # LSTM Gamma statistics
        if self.results['lstm_adaptive']['gamma_values']:
            gammas = self.results['lstm_adaptive']['gamma_values']
            print(f"\nLSTM Dynamic Gamma Statistics:")
            print(f"  Average: {np.mean(gammas):.3f}")
            print(f"  Std Dev: {np.std(gammas):.3f}")
            print(f"  Min: {min(gammas):.3f}")
            print(f"  Max: {max(gammas):.3f}")
    
    def save_results(self):
        """Save results to files."""
        # Save JSON results
        json_results = {}
        for model in ['sequential_supervised', 'lstm_adaptive']:
            json_results[model] = {
                k: v for k, v in self.results[model].items()
                if k not in ['trades', 'gamma_values']
            }
        json_results['metadata'] = self.results['metadata']
        
        with open(f'{self.output_dir}/results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save detailed report
        with open(f'{self.output_dir}/detailed_report.txt', 'w') as f:
            f.write("FULL DATASET COMPARISON REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Date Range: {self.start_date} to {self.end_date}\n")
            f.write(f"Total Files: {len(self.all_files)}\n")
            f.write(f"Training Files: {self.results['metadata']['train_files']}\n")
            f.write(f"Validation Files: {self.results['metadata']['val_files']}\n")
            f.write(f"Testing Files: {self.results['metadata']['test_files']}\n\n")
            
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*40 + "\n\n")
            
            for model_name in ['sequential_supervised', 'lstm_adaptive']:
                f.write(f"{model_name.upper().replace('_', ' ')}\n")
                if 'statistics' in self.results[model_name]:
                    stats = self.results[model_name]['statistics']
                    f.write(f"  Total Return: {stats['total_return']*100:.2f}%\n")
                    f.write(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}\n")
                    f.write(f"  Win Rate: {stats['win_rate']*100:.1f}%\n")
                    f.write(f"  Max Drawdown: {stats['max_drawdown']*100:.1f}%\n")
                    f.write(f"  Avg Monthly: {stats['avg_monthly_return']*100:.2f}%\n")
                    f.write(f"  Std Monthly: {stats['std_monthly_return']*100:.2f}%\n\n")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """Run the full dataset comparison."""
    comparison = FullDatasetComparison()
    results = comparison.run_full_comparison()
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    # Clean up memory
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results


if __name__ == "__main__":
    results = main()