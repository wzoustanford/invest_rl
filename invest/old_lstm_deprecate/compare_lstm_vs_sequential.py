"""
Side-by-side comparison of LSTM vs Sequential Supervised models
Compares monthly trading performance with:
- Sequential Supervised: Fixed gamma = 0.5
- LSTM: Dynamic gamma prediction
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

from sequential_supervised_trainer import SequentialSupervisedTrainer, TrainingConfig
from model.lstm_adaptive_model import LSTMAdaptiveModel


class ModelComparison:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data paths
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data files
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files")
        
        # Model configurations
        self.sequence_length = 7  # Use 7 consecutive days
        self.holding_period = 25  # 25-day holding period
        self.feature_dim = 240  # Fixed feature dimension
        
        # Results storage
        self.results = {
            'sequential_supervised': {
                'gamma': 0.5,
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
            }
        }
    
    def load_data_file(self, file_path):
        """Load and preprocess a data file."""
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def prepare_sequential_data(self, file_indices):
        """Prepare data for sequential supervised model."""
        all_features = []
        all_returns = []
        
        for idx in file_indices:
            data = self.load_data_file(self.all_files[idx])
            if data is None:
                continue
            
            # Extract features and returns
            if 'trainFeature' in data and data['trainFeature'] is not None:
                features = data['trainFeature']
                if 'train_in_portfolio_series' in data and data['train_in_portfolio_series'] is not None:
                    series = data['train_in_portfolio_series']
                    returns = (series[:, -1] - series[:, 0]) / (series[:, 0] + 1e-10)
                    
                    all_features.append(features)
                    all_returns.append(returns)
        
        if len(all_features) == len(file_indices):
            return all_features, all_returns
        return None, None
    
    def train_sequential_supervised(self, train_indices, gamma=0.5):
        """Train the sequential supervised model."""
        print(f"\nTraining Sequential Supervised Model (gamma={gamma})...")
        
        # Prepare training data
        features_list, returns_list = self.prepare_sequential_data(train_indices)
        if features_list is None:
            print("Failed to prepare training data")
            return None
        
        # Configure and train model
        config = TrainingConfig(
            gamma=gamma,
            learning_rate=0.001,
            num_steps=500,  # Reduced for faster training
            num_consecutive_days=len(train_indices),
            device=str(self.device)
        )
        
        trainer = SequentialSupervisedTrainer(config)
        
        # Find minimum number of stocks and features across all days
        min_stocks = min(f.shape[0] for f in features_list)
        min_features = min(f.shape[1] for f in features_list)
        
        # Truncate to consistent dimensions
        truncated_features = []
        truncated_returns = []
        for features, returns in zip(features_list, returns_list):
            truncated_features.append(features[:min_stocks, :min_features])
            truncated_returns.append(returns[:min_stocks])
        
        # Stack features for consecutive days
        stacked_features = torch.stack(truncated_features).to(self.device)
        stacked_returns = torch.stack(truncated_returns).to(self.device)
        
        # Train the model
        for step in range(config.num_steps):
            trainer.model.train()
            trainer.optimizer.zero_grad()
            
            # Forward pass through all consecutive days
            portfolio_returns_list = []
            for day_idx in range(len(features_list)):
                weights = trainer.model(stacked_features[day_idx])
                day_returns = torch.sum(weights * stacked_returns[day_idx])
                portfolio_returns_list.append(day_returns)
            
            # Calculate discounted cumulative return
            cumulative_return = 0
            gamma_power = 1.0
            for ret in portfolio_returns_list:
                cumulative_return += gamma_power * ret
                gamma_power *= gamma
            
            # Loss is negative return (we want to maximize)
            loss = -cumulative_return
            loss.backward()
            trainer.optimizer.step()
            
            if (step + 1) % 100 == 0:
                print(f"  Step {step+1}/{config.num_steps}, Loss: {loss.item():.4f}")
        
        return trainer.model
    
    def train_lstm_adaptive(self, train_indices):
        """Train the LSTM adaptive model."""
        print(f"\nTraining LSTM Adaptive Model...")
        
        # Prepare sequences
        sequences = []
        targets = []
        
        for i in range(len(train_indices) - self.sequence_length):
            seq_features = []
            valid = True
            
            # Get sequence of features
            for j in range(self.sequence_length):
                data = self.load_data_file(self.all_files[train_indices[i+j]])
                if data is None or 'trainFeature' not in data:
                    valid = False
                    break
                
                features = data['trainFeature'].cpu().numpy()
                # Pad/truncate to fixed dimension
                if features.shape[1] < self.feature_dim:
                    padding = np.zeros((features.shape[0], self.feature_dim - features.shape[1]))
                    features = np.concatenate([features, padding], axis=1)
                elif features.shape[1] > self.feature_dim:
                    features = features[:, :self.feature_dim]
                
                avg_features = np.mean(features, axis=0)
                seq_features.append(avg_features)
            
            if valid:
                # Get target returns
                target_data = self.load_data_file(self.all_files[train_indices[i + self.sequence_length]])
                if target_data and 'train_in_portfolio_series' in target_data:
                    series = target_data['train_in_portfolio_series'].cpu().numpy()
                    returns = (series[:, -1] - series[:, 0]) / (series[:, 0] + 1e-10)
                    
                    sequences.append(np.array(seq_features))
                    targets.append(returns[:100])  # Use top 100 stocks
        
        if len(sequences) == 0:
            print("Failed to prepare LSTM training data")
            return None
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        # Create and train LSTM model
        model = LSTMAdaptiveModel(
            input_dim=self.feature_dim,
            lstm_hidden_dim=64,
            lstm_layers=2,
            dropout_rate=0.2,
            num_stocks=targets.shape[1],
            sequence_length=self.sequence_length
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 50
        batch_size = 16
        
        for epoch in range(num_epochs):
            model.train()
            epoch_losses = []
            
            indices = np.random.permutation(len(sequences))
            
            for i in range(0, len(indices), batch_size):
                batch_idx = indices[i:i+batch_size]
                batch_seq = torch.FloatTensor(sequences[batch_idx]).to(self.device)
                batch_targets = torch.FloatTensor(targets[batch_idx]).to(self.device)
                
                optimizer.zero_grad()
                
                outputs = model(batch_seq, return_all_heads=True)
                portfolio_weights = outputs['portfolio_weights']
                portfolio_returns = torch.sum(portfolio_weights * batch_targets, dim=-1)
                
                # Sharpe ratio loss
                sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
                loss = -sharpe
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {np.mean(epoch_losses):.4f}")
        
        return model
    
    def backtest_month(self, model, model_type, test_indices):
        """Backtest a model for one month."""
        monthly_return = 0
        trades_count = 0
        
        for idx in test_indices:
            data = self.load_data_file(self.all_files[idx])
            if data is None:
                continue
            
            # Get test returns
            if 'test_in_portfolio_series' in data and data['test_in_portfolio_series'] is not None:
                series = data['test_in_portfolio_series'].cpu().numpy()
                actual_returns = (series[:, -1] - series[:, 0]) / (series[:, 0] + 1e-10)
                
                if model_type == 'sequential':
                    # Sequential supervised prediction
                    if 'testFeature' in data:
                        features = data['testFeature'].to(self.device)
                        with torch.no_grad():
                            weights = model(features).cpu().numpy()
                        
                        # Calculate portfolio return
                        num_stocks = min(len(weights), len(actual_returns))
                        weights = weights[:num_stocks]
                        actual_returns = actual_returns[:num_stocks]
                        portfolio_return = np.sum(weights * actual_returns)
                        
                        monthly_return += portfolio_return
                        trades_count += 1
                
                elif model_type == 'lstm':
                    # LSTM prediction (need sequence)
                    if idx >= self.sequence_length:
                        seq_features = []
                        valid = True
                        
                        for j in range(self.sequence_length):
                            seq_data = self.load_data_file(self.all_files[idx - self.sequence_length + j])
                            if seq_data and 'testFeature' in seq_data:
                                features = seq_data['testFeature'].cpu().numpy()
                                # Pad/truncate
                                if features.shape[1] < self.feature_dim:
                                    padding = np.zeros((features.shape[0], self.feature_dim - features.shape[1]))
                                    features = np.concatenate([features, padding], axis=1)
                                elif features.shape[1] > self.feature_dim:
                                    features = features[:, :self.feature_dim]
                                avg_features = np.mean(features, axis=0)
                                seq_features.append(avg_features)
                            else:
                                valid = False
                                break
                        
                        if valid and len(seq_features) == self.sequence_length:
                            seq_tensor = torch.FloatTensor([seq_features]).to(self.device)
                            with torch.no_grad():
                                outputs = model(seq_tensor, return_all_heads=True)
                                weights = outputs['portfolio_weights'].cpu().numpy()[0]
                                gamma_value = outputs['gamma_value'].cpu().item()
                            
                            # Calculate portfolio return
                            num_stocks = min(len(weights), len(actual_returns))
                            weights = weights[:num_stocks]
                            actual_returns = actual_returns[:num_stocks]
                            portfolio_return = np.sum(weights * actual_returns)
                            
                            monthly_return += portfolio_return
                            trades_count += 1
                            
                            # Store gamma value for LSTM
                            if model_type == 'lstm':
                                self.results['lstm_adaptive']['gamma_values'].append(gamma_value)
        
        if trades_count > 0:
            monthly_return /= trades_count
        
        return monthly_return - 0.0015  # Subtract transaction cost
    
    def run_comparison(self, start_year=2023, end_year=2024):
        """Run the complete comparison experiment."""
        print("\n" + "="*80)
        print("LSTM vs SEQUENTIAL SUPERVISED MODEL COMPARISON")
        print(f"Period: {start_year}-{end_year}")
        print("="*80)
        
        # Find files for the testing period
        test_start_idx = None
        test_end_idx = None
        
        for i, file_path in enumerate(self.all_files):
            if f'{start_year}_' in file_path and test_start_idx is None:
                test_start_idx = i
            if f'{end_year}_' in file_path:
                test_end_idx = i
        
        if test_start_idx is None or test_end_idx is None:
            print("Could not find test period in data")
            return
        
        # Use files before test period for training
        train_indices = list(range(max(0, test_start_idx - 100), test_start_idx))
        
        print(f"\nTraining on {len(train_indices)} files")
        print(f"Testing on files {test_start_idx} to {test_end_idx}")
        
        # Train both models
        seq_model = self.train_sequential_supervised(train_indices[-self.sequence_length:], gamma=0.5)
        lstm_model = self.train_lstm_adaptive(train_indices)
        
        if seq_model is None or lstm_model is None:
            print("Failed to train models")
            return
        
        # Monthly backtesting
        print("\n" + "="*60)
        print("MONTHLY BACKTESTING")
        print("="*60)
        
        months = []
        seq_monthly_returns = []
        lstm_monthly_returns = []
        seq_cumulative = 1.0
        lstm_cumulative = 1.0
        
        # Process each month
        current_month_start = test_start_idx
        month_size = 20  # Approximately 20 trading days per month
        month_num = 1
        
        while current_month_start < test_end_idx:
            current_month_end = min(current_month_start + month_size, test_end_idx)
            month_indices = list(range(current_month_start, current_month_end))
            
            # Backtest sequential model
            seq_return = self.backtest_month(seq_model, 'sequential', month_indices)
            seq_cumulative *= (1 + seq_return)
            seq_monthly_returns.append(seq_return)
            
            # Backtest LSTM model
            lstm_return = self.backtest_month(lstm_model, 'lstm', month_indices)
            lstm_cumulative *= (1 + lstm_return)
            lstm_monthly_returns.append(lstm_return)
            
            # Store results
            month_label = f"Month {month_num}"
            months.append(month_label)
            
            print(f"\n{month_label}:")
            print(f"  Sequential (γ=0.5): {seq_return*100:>7.2f}% | Cumulative: {(seq_cumulative-1)*100:>7.2f}%")
            print(f"  LSTM (dynamic γ):   {lstm_return*100:>7.2f}% | Cumulative: {(lstm_cumulative-1)*100:>7.2f}%")
            
            self.results['sequential_supervised']['monthly_returns'].append(seq_return)
            self.results['sequential_supervised']['cumulative_returns'].append(seq_cumulative - 1)
            self.results['lstm_adaptive']['monthly_returns'].append(lstm_return)
            self.results['lstm_adaptive']['cumulative_returns'].append(lstm_cumulative - 1)
            
            current_month_start = current_month_end
            month_num += 1
        
        # Calculate statistics
        self.calculate_statistics()
        
        # Print summary
        self.print_summary()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def calculate_statistics(self):
        """Calculate performance statistics for both models."""
        for model_name in ['sequential_supervised', 'lstm_adaptive']:
            returns = self.results[model_name]['monthly_returns']
            if len(returns) > 0:
                self.results[model_name]['statistics'] = {
                    'total_return': self.results[model_name]['cumulative_returns'][-1] if self.results[model_name]['cumulative_returns'] else 0,
                    'avg_monthly_return': np.mean(returns),
                    'std_monthly_return': np.std(returns),
                    'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(12),  # Annualized
                    'max_monthly_return': max(returns),
                    'min_monthly_return': min(returns),
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                    'num_months': len(returns)
                }
    
    def print_summary(self):
        """Print comparison summary."""
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        # Header
        print(f"\n{'Metric':<25} | {'Sequential (γ=0.5)':>20} | {'LSTM (dynamic γ)':>20}")
        print("-"*70)
        
        # Performance metrics
        metrics = [
            ('Total Return (%)', 'total_return', 100),
            ('Avg Monthly Return (%)', 'avg_monthly_return', 100),
            ('Std Monthly Return (%)', 'std_monthly_return', 100),
            ('Sharpe Ratio', 'sharpe_ratio', 1),
            ('Win Rate (%)', 'win_rate', 100),
            ('Best Month (%)', 'max_monthly_return', 100),
            ('Worst Month (%)', 'min_monthly_return', 100)
        ]
        
        for label, key, multiplier in metrics:
            seq_val = self.results['sequential_supervised']['statistics'].get(key, 0) * multiplier
            lstm_val = self.results['lstm_adaptive']['statistics'].get(key, 0) * multiplier
            print(f"{label:<25} | {seq_val:>20.2f} | {lstm_val:>20.2f}")
        
        # Gamma statistics for LSTM
        if self.results['lstm_adaptive']['gamma_values']:
            avg_gamma = np.mean(self.results['lstm_adaptive']['gamma_values'])
            std_gamma = np.std(self.results['lstm_adaptive']['gamma_values'])
            print(f"\nLSTM Gamma Statistics:")
            print(f"  Average: {avg_gamma:.3f}")
            print(f"  Std Dev: {std_gamma:.3f}")
            print(f"  Min: {min(self.results['lstm_adaptive']['gamma_values']):.3f}")
            print(f"  Max: {max(self.results['lstm_adaptive']['gamma_values']):.3f}")
    
    def save_results(self):
        """Save comparison results to files."""
        # Save JSON results
        json_results = {
            'sequential_supervised': {
                k: v for k, v in self.results['sequential_supervised'].items() 
                if k != 'trades'
            },
            'lstm_adaptive': {
                k: v for k, v in self.results['lstm_adaptive'].items() 
                if k not in ['trades', 'gamma_values']
            }
        }
        
        with open(f'{self.output_dir}/comparison_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save detailed report
        with open(f'{self.output_dir}/comparison_report.txt', 'w') as f:
            f.write("LSTM vs SEQUENTIAL SUPERVISED MODEL COMPARISON\n")
            f.write("="*60 + "\n\n")
            
            # Monthly returns table
            f.write("MONTHLY RETURNS\n")
            f.write("-"*40 + "\n")
            f.write(f"{'Month':<10} | {'Sequential':>12} | {'LSTM':>12}\n")
            f.write("-"*40 + "\n")
            
            for i in range(len(self.results['sequential_supervised']['monthly_returns'])):
                seq_ret = self.results['sequential_supervised']['monthly_returns'][i] * 100
                lstm_ret = self.results['lstm_adaptive']['monthly_returns'][i] * 100
                f.write(f"Month {i+1:<4} | {seq_ret:>11.2f}% | {lstm_ret:>11.2f}%\n")
            
            f.write("\n" + "="*60 + "\n")
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*40 + "\n\n")
            
            for model_name in ['sequential_supervised', 'lstm_adaptive']:
                f.write(f"{model_name.upper().replace('_', ' ')}\n")
                stats = self.results[model_name]['statistics']
                f.write(f"  Total Return: {stats['total_return']*100:.2f}%\n")
                f.write(f"  Sharpe Ratio: {stats['sharpe_ratio']:.3f}\n")
                f.write(f"  Win Rate: {stats['win_rate']*100:.1f}%\n")
                f.write(f"  Avg Monthly: {stats['avg_monthly_return']*100:.2f}%\n")
                f.write(f"  Std Monthly: {stats['std_monthly_return']*100:.2f}%\n")
                f.write("\n")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """Run the comparison experiment."""
    comparison = ModelComparison()
    results = comparison.run_comparison(start_year=2023, end_year=2024)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()