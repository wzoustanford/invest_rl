"""
LSTM Adaptive Trading Experiment
Addresses inconsistency issues with:
1. LSTM for temporal smoothing
2. Dynamic gamma prediction
3. Trade timing decisions
4. Ensemble averaging for stability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
import time
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from model.lstm_adaptive_model import LSTMAdaptiveModel, LSTMEnsemble


class LSTMAdaptiveExperiment:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # Data paths
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/lstm_adaptive_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load file list
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} files")
        print(f"Output directory: {self.output_dir}")
        
        # Model parameters
        self.sequence_length = 7  # Use 7 days initially for faster training
        self.lstm_hidden_dim = 64  # Smaller model for faster iteration
        self.lstm_layers = 2
        self.dropout_rate = 0.2
        
        # Training parameters
        self.batch_size = 16  # Smaller batch for limited data
        self.learning_rate = 0.001
        self.num_epochs = 50  # Fewer epochs for faster testing
        self.patience = 10  # Early stopping patience
        
        # Initialize results storage
        self.results = {
            'baseline': {},
            'lstm_single': {},
            'lstm_ensemble': {},
            'lstm_adaptive': {}
        }
    
    def prepare_sequence_data(self, file_indices, sequence_length=14):
        """
        Prepare sequences of data for LSTM training
        
        Args:
            file_indices: List of file indices to use
            sequence_length: Number of timesteps in sequence
            
        Returns:
            Dictionary with prepared sequences
        """
        sequences = []
        targets = []
        metadata = []
        
        # Need at least sequence_length + 1 files
        if len(file_indices) <= sequence_length:
            print(f"  Not enough files: {len(file_indices)} <= {sequence_length}")
            return None
        
        for i in range(len(file_indices) - sequence_length):
            # Get sequence of files
            seq_files = [self.all_files[file_indices[i+j]] for j in range(sequence_length)]
            target_file = self.all_files[file_indices[i + sequence_length]]
            
            # Load and process sequence
            seq_features = []
            seq_returns = []
            valid_sequence = True
            
            for file_path in seq_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            data = pickle.load(f)
                            
                        # Extract features (using training data)
                        if 'trainFeature' in data and data['trainFeature'] is not None:
                            features = data['trainFeature'].cpu().numpy()
                            # Average across stocks for timestep representation
                            avg_features = np.mean(features, axis=0)
                            seq_features.append(avg_features)
                            
                            # Get returns if available
                            if 'train_in_portfolio_series' in data:
                                series = data['train_in_portfolio_series'].cpu().numpy()
                                returns = (series[:, -1] - series[:, 0]) / (series[:, 0] + 1e-10)
                                seq_returns.append(returns)
                        else:
                            valid_sequence = False
                            break
                    except Exception as e:
                        valid_sequence = False
                        break
                else:
                    valid_sequence = False
                    break
            
            if valid_sequence and len(seq_features) == sequence_length:
                # Load target data
                try:
                    with open(target_file, 'rb') as f:
                        target_data = pickle.load(f)
                    
                    if 'test_in_portfolio_series' in target_data and target_data['test_in_portfolio_series'] is not None:
                        target_series = target_data['test_in_portfolio_series'].cpu().numpy()
                        target_returns = (target_series[:, -1] - target_series[:, 0]) / (target_series[:, 0] + 1e-10)
                        
                        sequences.append(np.array(seq_features))
                        targets.append(target_returns)
                        metadata.append({
                            'target_file': target_file,
                            'sequence_files': seq_files,
                            'historical_returns': seq_returns
                        })
                except:
                    pass
        
        print(f"Prepared {len(sequences)} sequences from {len(file_indices)} files")
        
        if len(sequences) > 0:
            return {
                'sequences': np.array(sequences),
                'targets': np.array(targets),
                'metadata': metadata
            }
        else:
            return None
    
    def calculate_optimal_gamma(self, historical_returns, lookahead_window=5):
        """
        Calculate optimal gamma based on recent performance
        Uses historical returns to determine best gamma value
        """
        if len(historical_returns) < lookahead_window:
            return 0.3  # Default
        
        recent_returns = historical_returns[-lookahead_window:]
        
        # Calculate volatility
        volatility = np.std(recent_returns)
        
        # Calculate trend
        trend = np.mean(recent_returns)
        
        # Adaptive gamma based on market conditions
        if volatility > 0.1:  # High volatility
            optimal_gamma = 0.5  # Higher gamma for risk management
        elif trend > 0.02:  # Strong positive trend
            optimal_gamma = 0.3  # Medium gamma for trend following
        elif trend < -0.02:  # Strong negative trend
            optimal_gamma = 0.1  # Lower gamma for quick adaptation
        else:  # Sideways market
            optimal_gamma = 0.3  # Default
        
        return optimal_gamma
    
    def should_trade_decision(self, predictions, confidence_threshold=0.6):
        """
        Decide whether to trade based on model confidence
        """
        portfolio_conf = predictions.get('portfolio_confidence', 0.5)
        gamma_conf = predictions.get('gamma_confidence', 0.5)
        
        # Trade if confidence is high enough
        if portfolio_conf > confidence_threshold and gamma_conf > confidence_threshold:
            return True
        
        # Also check if model explicitly recommends trading
        if 'should_trade' in predictions:
            return predictions['should_trade'].item() if torch.is_tensor(predictions['should_trade']) else predictions['should_trade']
        
        return portfolio_conf > confidence_threshold
    
    def train_lstm_model(self, train_data, val_data=None, model_type='single'):
        """
        Train LSTM model with early stopping
        
        Args:
            train_data: Training data dictionary
            val_data: Validation data dictionary
            model_type: 'single' or 'ensemble'
        """
        print(f"\nTraining LSTM model (type: {model_type})...")
        
        # Get dimensions
        input_dim = train_data['sequences'].shape[2]
        num_stocks = train_data['targets'].shape[1]
        
        # Create model
        if model_type == 'ensemble':
            model = LSTMEnsemble(
                num_models=3,
                input_dim=input_dim,
                lstm_hidden_dims=[64, 128, 256],
                num_stocks=num_stocks
            )
        else:
            model = LSTMAdaptiveModel(
                input_dim=input_dim,
                lstm_hidden_dim=self.lstm_hidden_dim,
                lstm_layers=self.lstm_layers,
                dropout_rate=self.dropout_rate,
                num_stocks=num_stocks,
                sequence_length=self.sequence_length
            )
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.num_epochs):
            # Training phase
            model.train()
            epoch_losses = []
            
            # Create mini-batches
            indices = np.random.permutation(len(train_data['sequences']))
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                
                # Get batch data
                batch_seq = torch.FloatTensor(train_data['sequences'][batch_idx]).to(self.device)
                batch_targets = torch.FloatTensor(train_data['targets'][batch_idx]).to(self.device)
                
                # Calculate optimal gamma for each sample
                batch_gammas = []
                for idx in batch_idx:
                    hist_returns = train_data['metadata'][idx].get('historical_returns', [])
                    if hist_returns:
                        # Use mean return across stocks for gamma calculation
                        mean_returns = [np.mean(r) for r in hist_returns[-5:]]
                        optimal_gamma = self.calculate_optimal_gamma(mean_returns)
                    else:
                        optimal_gamma = 0.3
                    batch_gammas.append(optimal_gamma)
                batch_gammas = torch.FloatTensor(batch_gammas).to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(batch_seq, return_all_heads=True)
                
                # Calculate portfolio returns
                portfolio_weights = outputs['portfolio_weights']
                portfolio_returns = torch.sum(portfolio_weights * batch_targets, dim=-1)
                
                # Sharpe ratio loss
                sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
                portfolio_loss = -sharpe
                
                # Gamma prediction loss
                if 'gamma_value' in outputs:
                    gamma_loss = nn.MSELoss()(outputs['gamma_value'], batch_gammas)
                else:
                    gamma_loss = 0
                
                # Total loss
                total_loss = portfolio_loss + 0.1 * gamma_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # Validation phase
            if val_data is not None:
                model.eval()
                val_loss = self.evaluate_model(model, val_data)
                val_losses.append(val_loss)
                
                # Learning rate scheduling
                scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(model.state_dict(), f'{self.output_dir}/best_model_{model_type}.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            train_losses.append(np.mean(epoch_losses))
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {train_losses[-1]:.4f}", end='')
                if val_data is not None:
                    print(f", Val Loss: {val_losses[-1]:.4f}")
                else:
                    print()
        
        # Load best model if validation was used
        if val_data is not None and os.path.exists(f'{self.output_dir}/best_model_{model_type}.pth'):
            model.load_state_dict(torch.load(f'{self.output_dir}/best_model_{model_type}.pth'))
        
        return model, train_losses, val_losses
    
    def evaluate_model(self, model, data):
        """
        Evaluate model on data
        """
        model.eval()
        losses = []
        
        with torch.no_grad():
            for i in range(0, len(data['sequences']), self.batch_size):
                batch_seq = torch.FloatTensor(
                    data['sequences'][i:i+self.batch_size]
                ).to(self.device)
                batch_targets = torch.FloatTensor(
                    data['targets'][i:i+self.batch_size]
                ).to(self.device)
                
                outputs = model(batch_seq, return_all_heads=True)
                portfolio_weights = outputs['portfolio_weights']
                portfolio_returns = torch.sum(portfolio_weights * batch_targets, dim=-1)
                sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
                loss = -sharpe
                
                losses.append(loss.item())
        
        return np.mean(losses)
    
    def backtest_strategy(self, model, test_files, strategy_name='lstm'):
        """
        Backtest the LSTM strategy
        """
        print(f"\nBacktesting {strategy_name} strategy...")
        
        cumulative_return = 1.0
        monthly_returns = []
        trades_executed = []
        trades_skipped = []
        
        for i, test_file in enumerate(test_files):
            # Prepare sequence for this test file
            file_idx = self.all_files.index(test_file)
            if file_idx < self.sequence_length:
                continue
            
            # Get sequence of previous files
            seq_data = self.prepare_sequence_data(
                list(range(file_idx - self.sequence_length, file_idx + 1)),
                self.sequence_length
            )
            
            if seq_data is None or len(seq_data['sequences']) == 0:
                continue
            
            # Make prediction
            model.eval()
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(seq_data['sequences'][-1:]).to(self.device)
                predictions = model(seq_tensor, return_all_heads=True)
                
                # Get portfolio weights
                portfolio_weights = predictions['portfolio_weights'].cpu().numpy()[0]
                
                # Get gamma value
                if 'gamma_value' in predictions:
                    gamma = predictions['gamma_value'].cpu().item()
                else:
                    gamma = 0.3
                
                # Check if should trade
                should_trade = self.should_trade_decision(predictions)
            
            if should_trade:
                # Execute trade
                try:
                    with open(test_file, 'rb') as f:
                        test_data = pickle.load(f)
                    
                    if 'test_in_portfolio_series' in test_data and test_data['test_in_portfolio_series'] is not None:
                        series = test_data['test_in_portfolio_series'].cpu().numpy()
                        returns = (series[:, -1] - series[:, 0]) / (series[:, 0] + 1e-10)
                        
                        # Calculate portfolio return
                        portfolio_return = np.sum(portfolio_weights * returns) - 0.0015  # Transaction cost
                        
                        cumulative_return *= (1 + portfolio_return)
                        monthly_returns.append(portfolio_return)
                        
                        trades_executed.append({
                            'file': test_file,
                            'return': portfolio_return,
                            'gamma': gamma,
                            'cumulative': cumulative_return
                        })
                        
                        print(f"  Trade {i+1}: Return={portfolio_return*100:.2f}%, Gamma={gamma:.2f}, Cumulative={(cumulative_return-1)*100:.2f}%")
                except Exception as e:
                    print(f"  Error processing {test_file}: {e}")
            else:
                trades_skipped.append({
                    'file': test_file,
                    'reason': 'Low confidence'
                })
                print(f"  Trade {i+1}: SKIPPED (low confidence)")
        
        # Calculate statistics
        if monthly_returns:
            stats = {
                'total_return': cumulative_return - 1,
                'num_trades': len(trades_executed),
                'num_skipped': len(trades_skipped),
                'avg_return': np.mean(monthly_returns),
                'std_return': np.std(monthly_returns),
                'win_rate': sum(1 for r in monthly_returns if r > 0) / len(monthly_returns),
                'sharpe': np.mean(monthly_returns) / (np.std(monthly_returns) + 1e-8),
                'max_return': max(monthly_returns),
                'min_return': min(monthly_returns),
                'trades': trades_executed
            }
        else:
            stats = {'total_return': 0, 'num_trades': 0}
        
        return stats
    
    def run_experiment(self, start_year=2021, end_year=2022):
        """
        Run the complete LSTM experiment
        """
        print(f"\n{'='*80}")
        print(f"LSTM ADAPTIVE TRADING EXPERIMENT")
        print(f"Testing Period: {start_year}-{end_year}")
        print(f"{'='*80}")
        
        # Find files for the testing period
        test_files = []
        train_files = []
        
        for file_path in self.all_files:
            # Extract date from filename
            if 'test_data_start_date_' in file_path:
                try:
                    date_str = file_path.split('test_data_start_date_')[1].split('_news')[0]
                    date_obj = datetime.strptime(date_str, '%Y_%m_%d')
                    actual_trading_date = date_obj + timedelta(days=360)
                    
                    if start_year <= actual_trading_date.year <= end_year:
                        test_files.append(file_path)
                    elif actual_trading_date.year < start_year:
                        train_files.append(file_path)
                except:
                    continue
        
        print(f"Found {len(train_files)} training files")
        print(f"Found {len(test_files)} test files")
        
        # If no explicit training files, use early test files for training
        if len(train_files) == 0:
            print("No pre-period training files found. Using first 100 files for training.")
            train_files = self.all_files[:100]
            train_indices = list(range(100))
        else:
            train_indices = [self.all_files.index(f) for f in train_files if f in self.all_files]
        
        # Prepare training data
        print("\nPreparing training sequences...")
        train_data = self.prepare_sequence_data(train_indices, self.sequence_length)
        
        if train_data is None or len(train_data['sequences']) == 0:
            print("Failed to prepare training data")
            return None
        
        # Split training data for validation
        n_train = int(0.8 * len(train_data['sequences']))
        val_data = {
            'sequences': train_data['sequences'][n_train:],
            'targets': train_data['targets'][n_train:],
            'metadata': train_data['metadata'][n_train:]
        }
        train_data = {
            'sequences': train_data['sequences'][:n_train],
            'targets': train_data['targets'][:n_train],
            'metadata': train_data['metadata'][:n_train]
        }
        
        print(f"Training samples: {len(train_data['sequences'])}")
        print(f"Validation samples: {len(val_data['sequences'])}")
        
        # Train single LSTM model
        print("\n" + "="*60)
        print("Training Single LSTM Model")
        print("="*60)
        lstm_model, train_losses, val_losses = self.train_lstm_model(
            train_data, val_data, model_type='single'
        )
        
        # Train ensemble LSTM model
        print("\n" + "="*60)
        print("Training Ensemble LSTM Model")
        print("="*60)
        ensemble_model, ensemble_train_losses, ensemble_val_losses = self.train_lstm_model(
            train_data, val_data, model_type='ensemble'
        )
        
        # Backtest strategies
        print("\n" + "="*60)
        print("BACKTESTING RESULTS")
        print("="*60)
        
        # Test single LSTM
        lstm_results = self.backtest_strategy(lstm_model, test_files[:24], 'LSTM Single')
        self.results['lstm_single'] = lstm_results
        
        # Test ensemble LSTM
        ensemble_results = self.backtest_strategy(ensemble_model, test_files[:24], 'LSTM Ensemble')
        self.results['lstm_ensemble'] = ensemble_results
        
        # Print comparison
        self.print_results_comparison()
        
        # Save results
        self.save_results()
        
        return self.results
    
    def print_results_comparison(self):
        """
        Print comparison of all strategies
        """
        print("\n" + "="*80)
        print("STRATEGY COMPARISON")
        print("="*80)
        
        print(f"\n{'Strategy':<20} | {'Total Return':>12} | {'Trades':>8} | {'Win Rate':>10} | {'Sharpe':>10}")
        print("-"*80)
        
        for strategy, results in self.results.items():
            if 'total_return' in results:
                print(f"{strategy:<20} | {results['total_return']*100:>11.2f}% | "
                      f"{results.get('num_trades', 0):>8} | "
                      f"{results.get('win_rate', 0)*100:>9.1f}% | "
                      f"{results.get('sharpe', 0):>10.3f}")
    
    def save_results(self):
        """
        Save results to file
        """
        # Save detailed results
        with open(f'{self.output_dir}/results.json', 'w') as f:
            # Convert numpy types for JSON serialization
            results_json = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    results_json[key] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64, np.ndarray)) else v
                        for k, v in value.items() if k != 'trades'
                    }
            json.dump(results_json, f, indent=2)
        
        # Save summary report
        with open(f'{self.output_dir}/summary_report.txt', 'w') as f:
            f.write("LSTM ADAPTIVE TRADING EXPERIMENT RESULTS\n")
            f.write("="*60 + "\n\n")
            
            for strategy, results in self.results.items():
                if 'total_return' in results:
                    f.write(f"\n{strategy.upper()}\n")
                    f.write("-"*40 + "\n")
                    f.write(f"Total Return: {results['total_return']*100:.2f}%\n")
                    f.write(f"Number of Trades: {results.get('num_trades', 0)}\n")
                    f.write(f"Trades Skipped: {results.get('num_skipped', 0)}\n")
                    f.write(f"Win Rate: {results.get('win_rate', 0)*100:.1f}%\n")
                    f.write(f"Sharpe Ratio: {results.get('sharpe', 0):.3f}\n")
                    f.write(f"Best Trade: {results.get('max_return', 0)*100:.2f}%\n")
                    f.write(f"Worst Trade: {results.get('min_return', 0)*100:.2f}%\n")
        
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """
    Run the LSTM adaptive experiment
    """
    print("="*80)
    print("LSTM ADAPTIVE TRADING SYSTEM")
    print("Addressing inconsistency with temporal smoothing")
    print("="*80)
    
    experiment = LSTMAdaptiveExperiment()
    
    # Run initial experiment on 2021-2022 data
    results = experiment.run_experiment(start_year=2021, end_year=2022)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    # Print key insights
    print("\nKEY INSIGHTS:")
    print("-"*40)
    
    if results and 'lstm_ensemble' in results and 'total_return' in results['lstm_ensemble']:
        ensemble_return = results['lstm_ensemble']['total_return']
        if 'lstm_single' in results and 'total_return' in results['lstm_single']:
            single_return = results['lstm_single']['total_return']
            
            if ensemble_return > single_return:
                print(f"✓ Ensemble outperformed single model by {(ensemble_return-single_return)*100:.1f}%")
            else:
                print(f"✗ Single model outperformed ensemble by {(single_return-ensemble_return)*100:.1f}%")
        
        print(f"✓ Trade selectivity: {results['lstm_ensemble'].get('num_skipped', 0)} trades skipped for risk management")
        print(f"✓ Win rate: {results['lstm_ensemble'].get('win_rate', 0)*100:.1f}%")
    
    return results


if __name__ == "__main__":
    results = main()