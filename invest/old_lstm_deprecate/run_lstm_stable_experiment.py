"""
LSTM Trading Stability Experiment
Uses LSTM to:
1. Predict dynamic gamma values instead of fixed gamma
2. Decide when to trade vs when to wait
3. Stabilize trading results through temporal smoothing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from model.lstm_adaptive_model import LSTMAdaptiveModel, LSTMEnsemble


class LSTMStableTrading:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data paths
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/lstm_stable_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data file list
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Found {len(self.all_files)} data files")
        
        # Model configuration
        self.sequence_length = 10  # Use 10 trading periods for temporal context
        self.feature_dim = 240  # Fixed feature dimension (will pad/truncate as needed)
        self.lstm_hidden_dim = 128
        self.lstm_layers = 2
        self.dropout_rate = 0.2
        
        # Training configuration
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.early_stopping_patience = 15
        
        # Trading configuration
        self.transaction_cost = 0.0015  # 0.15% per trade
        self.confidence_threshold = 0.5  # Lowered threshold to allow more trades
        
        # Results storage
        self.results = {}
    
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess data from a pickle file
        Returns features and portfolio data, handling dimension mismatches
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract features
            if 'trainFeature' in data and data['trainFeature'] is not None:
                features = data['trainFeature'].cpu().numpy()
                
                # Handle dimension mismatch by padding or truncating
                current_dim = features.shape[1]
                if current_dim < self.feature_dim:
                    # Pad with zeros
                    padding = np.zeros((features.shape[0], self.feature_dim - current_dim))
                    features = np.concatenate([features, padding], axis=1)
                elif current_dim > self.feature_dim:
                    # Truncate
                    features = features[:, :self.feature_dim]
                
                # Get portfolio series for return calculation
                portfolio_series = None
                if 'train_in_portfolio_series' in data and data['train_in_portfolio_series'] is not None:
                    portfolio_series = data['train_in_portfolio_series'].cpu().numpy()
                
                return {
                    'features': features,
                    'portfolio_series': portfolio_series,
                    'num_stocks': features.shape[0]
                }
            
            # Try test data if training data not available
            elif 'testFeature' in data and data['testFeature'] is not None:
                features = data['testFeature'].cpu().numpy()
                
                # Handle dimension mismatch
                current_dim = features.shape[1]
                if current_dim < self.feature_dim:
                    padding = np.zeros((features.shape[0], self.feature_dim - current_dim))
                    features = np.concatenate([features, padding], axis=1)
                elif current_dim > self.feature_dim:
                    features = features[:, :self.feature_dim]
                
                portfolio_series = None
                if 'test_in_portfolio_series' in data and data['test_in_portfolio_series'] is not None:
                    portfolio_series = data['test_in_portfolio_series'].cpu().numpy()
                
                return {
                    'features': features,
                    'portfolio_series': portfolio_series,
                    'num_stocks': features.shape[0]
                }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
        
        return None
    
    def create_sequences(self, file_indices, for_training=True):
        """
        Create sequences for LSTM training/inference
        """
        sequences = []
        targets = []
        metadata = []
        
        for i in range(len(file_indices) - self.sequence_length):
            # Get sequence of files
            seq_data = []
            valid_sequence = True
            
            # Load sequence data
            for j in range(self.sequence_length):
                file_idx = file_indices[i + j]
                file_path = self.all_files[file_idx]
                data = self.load_and_preprocess_data(file_path)
                
                if data is None:
                    valid_sequence = False
                    break
                
                seq_data.append(data)
            
            if not valid_sequence or len(seq_data) != self.sequence_length:
                continue
            
            # Load target data (next period)
            target_idx = file_indices[i + self.sequence_length]
            target_path = self.all_files[target_idx]
            target_data = self.load_and_preprocess_data(target_path)
            
            if target_data is None or target_data['portfolio_series'] is None:
                continue
            
            # Create feature sequence (average across stocks for each timestep)
            seq_features = []
            for data in seq_data:
                avg_features = np.mean(data['features'], axis=0)
                seq_features.append(avg_features)
            
            # Calculate target returns
            target_series = target_data['portfolio_series']
            target_returns = (target_series[:, -1] - target_series[:, 0]) / (target_series[:, 0] + 1e-10)
            
            # Ensure consistent number of stocks (use top N by market cap/volume)
            num_stocks = min(100, len(target_returns))  # Use top 100 stocks
            target_returns = target_returns[:num_stocks]
            
            sequences.append(np.array(seq_features))
            targets.append(target_returns)
            metadata.append({
                'target_file': target_path,
                'sequence_indices': [file_indices[i+j] for j in range(self.sequence_length + 1)]
            })
        
        if len(sequences) > 0:
            return {
                'sequences': np.array(sequences),
                'targets': np.array(targets),
                'metadata': metadata
            }
        return None
    
    def calculate_dynamic_gamma(self, recent_returns, volatility):
        """
        Calculate adaptive gamma based on market conditions
        Higher gamma = more conservative (higher discount for future rewards)
        Lower gamma = more aggressive (value future rewards more)
        """
        # Base gamma
        base_gamma = 0.3
        
        # Adjust based on volatility (higher vol = higher gamma for safety)
        vol_adjustment = min(0.3, volatility * 2)
        
        # Adjust based on recent performance (negative returns = higher gamma)
        if recent_returns < -0.05:
            perf_adjustment = 0.2
        elif recent_returns < 0:
            perf_adjustment = 0.1
        elif recent_returns > 0.05:
            perf_adjustment = -0.1
        else:
            perf_adjustment = 0
        
        # Combine adjustments
        gamma = base_gamma + vol_adjustment + perf_adjustment
        
        # Clamp to valid range
        return np.clip(gamma, 0.1, 0.9)
    
    def train_lstm(self, train_sequences, val_sequences=None):
        """
        Train the LSTM model with adaptive gamma prediction
        """
        print("\n" + "="*60)
        print("TRAINING LSTM MODEL")
        print("="*60)
        
        # Model setup
        num_stocks = train_sequences['targets'].shape[1]
        model = LSTMAdaptiveModel(
            input_dim=self.feature_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers,
            dropout_rate=self.dropout_rate,
            num_stocks=num_stocks,
            sequence_length=self.sequence_length
        ).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            model.train()
            epoch_losses = []
            
            # Shuffle data
            indices = np.random.permutation(len(train_sequences['sequences']))
            
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i+self.batch_size]
                
                # Prepare batch
                batch_seq = torch.FloatTensor(train_sequences['sequences'][batch_idx]).to(self.device)
                batch_targets = torch.FloatTensor(train_sequences['targets'][batch_idx]).to(self.device)
                
                # Calculate target gamma values for supervision
                target_gammas = []
                for idx in batch_idx:
                    # Use historical returns to determine optimal gamma
                    if idx > 0:
                        prev_returns = train_sequences['targets'][idx-1]
                        mean_return = np.mean(prev_returns)
                        volatility = np.std(prev_returns)
                        optimal_gamma = self.calculate_dynamic_gamma(mean_return, volatility)
                    else:
                        optimal_gamma = 0.3
                    target_gammas.append(optimal_gamma)
                
                target_gammas = torch.FloatTensor(target_gammas).to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(batch_seq, return_all_heads=True)
                
                # Calculate losses
                portfolio_weights = outputs['portfolio_weights']
                portfolio_returns = torch.sum(portfolio_weights * batch_targets, dim=-1)
                
                # Sharpe ratio loss (negative for maximization)
                returns_mean = portfolio_returns.mean()
                returns_std = portfolio_returns.std() + 1e-8
                sharpe_loss = -returns_mean / returns_std
                
                # Gamma prediction loss
                gamma_loss = nn.MSELoss()(outputs['gamma_value'], target_gammas)
                
                # Timing loss (encourage trading when confidence is high)
                # Create pseudo-labels based on return expectations
                expected_returns = portfolio_returns.detach()
                should_trade = (expected_returns > 0).long()
                timing_loss = nn.CrossEntropyLoss()(outputs['timing_logits'], should_trade)
                
                # Combined loss
                total_loss = sharpe_loss + 0.2 * gamma_loss + 0.1 * timing_loss
                
                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_losses.append(total_loss.item())
            
            # Validation
            if val_sequences is not None:
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    for i in range(0, len(val_sequences['sequences']), self.batch_size):
                        batch_seq = torch.FloatTensor(
                            val_sequences['sequences'][i:i+self.batch_size]
                        ).to(self.device)
                        batch_targets = torch.FloatTensor(
                            val_sequences['targets'][i:i+self.batch_size]
                        ).to(self.device)
                        
                        outputs = model(batch_seq, return_all_heads=True)
                        portfolio_weights = outputs['portfolio_weights']
                        portfolio_returns = torch.sum(portfolio_weights * batch_targets, dim=-1)
                        
                        sharpe = portfolio_returns.mean() / (portfolio_returns.std() + 1e-8)
                        val_loss = -sharpe
                        val_losses.append(val_loss.item())
                
                avg_val_loss = np.mean(val_losses)
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f'{self.output_dir}/best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                avg_train_loss = np.mean(epoch_losses)
                print(f"Epoch {epoch+1}/{self.num_epochs}: Train Loss={avg_train_loss:.4f}", end='')
                if val_sequences is not None:
                    print(f", Val Loss={avg_val_loss:.4f}")
                else:
                    print()
        
        # Load best model if validation was used
        if val_sequences is not None and os.path.exists(f'{self.output_dir}/best_model.pth'):
            model.load_state_dict(torch.load(f'{self.output_dir}/best_model.pth'))
        
        return model
    
    def backtest_with_lstm(self, model, test_indices):
        """
        Backtest the LSTM strategy with dynamic gamma and trade timing
        """
        print("\n" + "="*60)
        print("BACKTESTING LSTM STRATEGY")
        print("="*60)
        
        model.eval()
        
        # Results tracking
        cumulative_return = 1.0
        trade_returns = []
        trades_executed = []
        trades_skipped = []
        gamma_values = []
        
        # Need at least sequence_length files before we can start trading
        for i in range(self.sequence_length, len(test_indices)):
            # Prepare sequence
            seq_indices = test_indices[i-self.sequence_length:i]
            seq_data = self.create_sequences(seq_indices + [test_indices[i]], for_training=False)
            
            if seq_data is None or len(seq_data['sequences']) == 0:
                continue
            
            # Get current test file for actual trading
            test_file = self.all_files[test_indices[i]]
            test_data = self.load_and_preprocess_data(test_file)
            
            if test_data is None or test_data['portfolio_series'] is None:
                continue
            
            # Make predictions
            with torch.no_grad():
                seq_tensor = torch.FloatTensor(seq_data['sequences'][-1:]).to(self.device)
                predictions = model(seq_tensor, return_all_heads=True)
                
                # Extract predictions
                portfolio_weights = predictions['portfolio_weights'].cpu().numpy()[0]
                gamma_value = predictions['gamma_value'].cpu().item()
                timing_probs = predictions['timing_probs'].cpu().numpy()[0]
                should_trade = timing_probs[0] > 0.5  # Index 0 is "trade" class
                trade_confidence = max(timing_probs)
            
            # Decide whether to trade (use OR condition for more trading)
            if (should_trade and trade_confidence > self.confidence_threshold) or gamma_value < 0.4:
                # Calculate actual returns
                portfolio_series = test_data['portfolio_series']
                actual_returns = (portfolio_series[:, -1] - portfolio_series[:, 0]) / (portfolio_series[:, 0] + 1e-10)
                
                # Ensure consistent number of stocks
                num_stocks = min(len(portfolio_weights), len(actual_returns))
                portfolio_weights = portfolio_weights[:num_stocks]
                actual_returns = actual_returns[:num_stocks]
                
                # Normalize weights
                portfolio_weights = portfolio_weights / (np.sum(portfolio_weights) + 1e-10)
                
                # Calculate portfolio return
                portfolio_return = np.sum(portfolio_weights * actual_returns) - self.transaction_cost
                
                # Update cumulative return
                cumulative_return *= (1 + portfolio_return)
                trade_returns.append(portfolio_return)
                gamma_values.append(gamma_value)
                
                trades_executed.append({
                    'index': i,
                    'file': test_file,
                    'return': portfolio_return,
                    'gamma': gamma_value,
                    'confidence': trade_confidence,
                    'cumulative': cumulative_return
                })
                
                print(f"  Period {i-self.sequence_length+1}: TRADE - Return={portfolio_return*100:.2f}%, "
                      f"Gamma={gamma_value:.3f}, Confidence={trade_confidence:.3f}, "
                      f"Cumulative={(cumulative_return-1)*100:.2f}%")
            else:
                trades_skipped.append({
                    'index': i,
                    'reason': 'Low confidence' if trade_confidence <= self.confidence_threshold else 'Model says wait',
                    'confidence': trade_confidence,
                    'gamma': gamma_value
                })
                print(f"  Period {i-self.sequence_length+1}: SKIP - Confidence={trade_confidence:.3f}, Gamma={gamma_value:.3f}")
        
        # Calculate statistics
        if trade_returns:
            stats = {
                'total_return': cumulative_return - 1,
                'num_trades': len(trades_executed),
                'num_skipped': len(trades_skipped),
                'trade_ratio': len(trades_executed) / (len(trades_executed) + len(trades_skipped)),
                'avg_return': np.mean(trade_returns),
                'std_return': np.std(trade_returns),
                'sharpe_ratio': np.mean(trade_returns) / (np.std(trade_returns) + 1e-8) * np.sqrt(252/25),  # Annualized
                'win_rate': sum(1 for r in trade_returns if r > 0) / len(trade_returns),
                'max_return': max(trade_returns),
                'min_return': min(trade_returns),
                'avg_gamma': np.mean(gamma_values),
                'std_gamma': np.std(gamma_values),
                'trades': trades_executed
            }
        else:
            stats = {
                'total_return': 0,
                'num_trades': 0,
                'num_skipped': len(trades_skipped),
                'message': 'No trades executed'
            }
        
        return stats
    
    def run_experiment(self):
        """
        Run the complete LSTM trading stability experiment
        """
        print("\n" + "="*80)
        print("LSTM TRADING STABILITY EXPERIMENT")
        print("Dynamic Gamma Prediction & Trade Timing")
        print("="*80)
        
        # Split data into train/validation/test
        # Use first 60% for training, next 20% for validation, last 20% for testing
        n_files = len(self.all_files)
        train_end = int(0.6 * n_files)
        val_end = int(0.8 * n_files)
        
        train_indices = list(range(train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, n_files))
        
        print(f"\nData split:")
        print(f"  Training: {len(train_indices)} files")
        print(f"  Validation: {len(val_indices)} files") 
        print(f"  Testing: {len(test_indices)} files")
        
        # Create sequences
        print("\nPreparing sequences...")
        train_sequences = self.create_sequences(train_indices)
        val_sequences = self.create_sequences(val_indices)
        
        if train_sequences is None:
            print("Failed to create training sequences")
            return None
        
        print(f"  Training sequences: {len(train_sequences['sequences'])}")
        if val_sequences:
            print(f"  Validation sequences: {len(val_sequences['sequences'])}")
        
        # Train LSTM model
        model = self.train_lstm(train_sequences, val_sequences)
        
        # Backtest on test data
        results = self.backtest_with_lstm(model, test_indices)
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results)
        
        return results
    
    def save_results(self, results):
        """
        Save experiment results
        """
        # Save JSON results
        json_results = {k: v for k, v in results.items() if k != 'trades'}
        with open(f'{self.output_dir}/results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save detailed report
        with open(f'{self.output_dir}/report.txt', 'w') as f:
            f.write("LSTM TRADING STABILITY EXPERIMENT RESULTS\n")
            f.write("="*60 + "\n\n")
            f.write("STRATEGY: Dynamic Gamma with Trade Timing\n")
            f.write("-"*40 + "\n\n")
            
            if results.get('num_trades', 0) > 0:
                f.write(f"Total Return: {results['total_return']*100:.2f}%\n")
                f.write(f"Number of Trades: {results['num_trades']}\n")
                f.write(f"Number Skipped: {results['num_skipped']}\n")
                f.write(f"Trade Execution Rate: {results['trade_ratio']*100:.1f}%\n")
                f.write(f"Win Rate: {results['win_rate']*100:.1f}%\n")
                f.write(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}\n")
                f.write(f"Average Return: {results['avg_return']*100:.2f}%\n")
                f.write(f"Return Volatility: {results['std_return']*100:.2f}%\n")
                f.write(f"Best Trade: {results['max_return']*100:.2f}%\n")
                f.write(f"Worst Trade: {results['min_return']*100:.2f}%\n")
                f.write(f"Average Gamma: {results['avg_gamma']:.3f}\n")
                f.write(f"Gamma Std Dev: {results['std_gamma']:.3f}\n")
            else:
                f.write("No trades were executed.\n")
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def print_summary(self, results):
        """
        Print experiment summary
        """
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        
        if results.get('num_trades', 0) > 0:
            print(f"\nKey Metrics:")
            print(f"  Total Return: {results['total_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"  Win Rate: {results['win_rate']*100:.1f}%")
            print(f"  Trades Executed: {results['num_trades']} / {results['num_trades'] + results['num_skipped']}")
            
            print(f"\nRisk Management:")
            print(f"  Trade Selectivity: {results['trade_ratio']*100:.1f}% of opportunities taken")
            print(f"  Average Gamma: {results['avg_gamma']:.3f} (adaptive discount factor)")
            print(f"  Return Volatility: {results['std_return']*100:.2f}%")
            
            print(f"\nTrading Stability Features:")
            print(f"  ✓ Dynamic gamma prediction based on market conditions")
            print(f"  ✓ Selective trading - skipped {results['num_skipped']} low-confidence trades")
            print(f"  ✓ Temporal smoothing using {self.sequence_length}-period sequences")
            
            if results['win_rate'] > 0.5:
                print(f"  ✓ Positive win rate indicates effective trade selection")
            if results['std_gamma'] > 0.05:
                print(f"  ✓ Variable gamma (std={results['std_gamma']:.3f}) shows adaptive behavior")
        else:
            print("\nNo trades executed - model was too conservative")
            print("Consider adjusting confidence threshold or training parameters")


def main():
    """
    Run the LSTM trading stability experiment
    """
    experiment = LSTMStableTrading()
    results = experiment.run_experiment()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    
    return results


if __name__ == "__main__":
    results = main()