"""
Simplified LSTM experiment focusing on consistency improvements
Uses existing Sequential Supervised model outputs with LSTM smoothing
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import os
import json
from datetime import datetime, timedelta
import gc

# Import the existing model
from model.iimodel import IIMODEL


class SimpleLSTM(nn.Module):
    """
    Simple LSTM that takes historical predictions and outputs smoothed weights
    """
    def __init__(self, input_dim=3, hidden_dim=32, num_layers=2, output_dim=3):
        super(SimpleLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Take last timestep
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return self.softmax(output)


class LSTMConsistencyExperiment:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/experiments/lstm_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load file list
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Files available: {len(self.all_files)}")
        
        # LSTM for gamma selection
        self.lstm_gamma = SimpleLSTM(input_dim=3, hidden_dim=32, output_dim=3).to(self.device)
        
        # Store historical results for LSTM input
        self.history_buffer = []
        
    def train_base_model(self, training_files, gamma=0.3):
        """Train base sequential model with given gamma"""
        
        # Load data sequence
        data_seq = []
        for f in training_files[-7:]:  # Use last 7 files
            if os.path.exists(f):
                try:
                    with open(f, 'rb') as file:
                        data = pickle.load(file)
                        data['trainFeature'] = data['trainFeature'].to(self.device)
                        data['train_in_portfolio_series'] = data['train_in_portfolio_series'].to(self.device)
                        data_seq.append(data)
                except:
                    return None
        
        if len(data_seq) != 7:
            return None
        
        # Initialize model
        model = IIMODEL(dropout_ratio=0.0, num_conv_filters=32, hidden_dim=47).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Quick training (100 iterations for speed)
        model.train()
        for step in range(100):
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            
            # Process sequence with gamma discounting
            for i in range(7):
                features = data_seq[i]['trainFeature']
                series = data_seq[i]['train_in_portfolio_series']
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                
                portfolio_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                daily_values = torch.sum(series * shares, dim=0)
                daily_returns = daily_values[1:] - daily_values[:-1]
                sharpe = portfolio_return / (torch.std(daily_returns) + 1e-10)
                
                gamma_power = gamma ** (7 - i - 1)
                loss = -sharpe * gamma_power
                total_loss = total_loss + loss
            
            total_loss.backward()
            optimizer.step()
        
        return model
    
    def evaluate_model(self, model, test_file):
        """Evaluate model on test file"""
        if not os.path.exists(test_file) or model is None:
            return None
        
        try:
            with open(test_file, 'rb') as f:
                test_data = pickle.load(f)
            
            if test_data.get('test_in_portfolio_series') is None:
                return None
            
            model.eval()
            with torch.no_grad():
                features = test_data['testFeature'].to(self.device)
                series = test_data['test_in_portfolio_series'].to(self.device)
                
                weights = model(features)
                shares = weights / (series[:, 0:1] + 1e-10)
                
                raw_return = torch.sum((series[:, -1:] - series[:, 0:1]) * shares)
                net_return = raw_return - 0.0015  # Transaction cost
                
                return net_return.item()
        except:
            return None
    
    def run_adaptive_strategy(self, year=2021):
        """Run strategy with LSTM-based gamma adaptation"""
        
        print(f"\n{'='*60}")
        print(f"LSTM ADAPTIVE STRATEGY - {year}")
        print(f"{'='*60}")
        
        # Find relevant files
        test_files = []
        for fpath in self.all_files:
            if 'test_data_start_date_' in fpath:
                try:
                    date_str = fpath.split('test_data_start_date_')[1].split('_news')[0]
                    date_obj = datetime.strptime(date_str, '%Y_%m_%d')
                    trading_date = date_obj + timedelta(days=360)
                    
                    if trading_date.year == year:
                        test_files.append(fpath)
                except:
                    continue
        
        print(f"Found {len(test_files)} files for {year}")
        
        results = {
            'fixed_gamma_0.1': [],
            'fixed_gamma_0.3': [],
            'fixed_gamma_0.5': [],
            'lstm_adaptive': []
        }
        
        # Test each month
        for i, test_file in enumerate(test_files[:12]):  # Max 12 months
            print(f"\nMonth {i+1}:")
            
            # Find training files (previous 7)
            file_idx = self.all_files.index(test_file)
            if file_idx < 7:
                continue
            
            training_files = self.all_files[file_idx-6:file_idx+1]
            
            # Test fixed gammas
            for gamma in [0.1, 0.3, 0.5]:
                model = self.train_base_model(training_files, gamma)
                ret = self.evaluate_model(model, test_file)
                
                if ret is not None:
                    results[f'fixed_gamma_{gamma}'].append(ret)
                    print(f"  γ={gamma}: {ret*100:+.2f}%")
                
                del model
                torch.cuda.empty_cache()
            
            # LSTM adaptive gamma selection
            if len(self.history_buffer) >= 3:
                # Use LSTM to predict best gamma
                history_tensor = torch.FloatTensor(self.history_buffer[-7:]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    gamma_probs = self.lstm_gamma(history_tensor)
                    best_gamma_idx = torch.argmax(gamma_probs).item()
                    adaptive_gamma = [0.1, 0.3, 0.5][best_gamma_idx]
            else:
                # Default to 0.3 initially
                adaptive_gamma = 0.3
            
            # Train with adaptive gamma
            model = self.train_base_model(training_files, adaptive_gamma)
            ret = self.evaluate_model(model, test_file)
            
            if ret is not None:
                results['lstm_adaptive'].append(ret)
                print(f"  LSTM (γ={adaptive_gamma}): {ret*100:+.2f}%")
                
                # Update history buffer
                self.history_buffer.append([
                    results['fixed_gamma_0.1'][-1] if results['fixed_gamma_0.1'] else 0,
                    results['fixed_gamma_0.3'][-1] if results['fixed_gamma_0.3'] else 0,
                    results['fixed_gamma_0.5'][-1] if results['fixed_gamma_0.5'] else 0
                ])
            
            del model
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate statistics
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY - {year}")
        print(f"{'='*60}")
        
        for strategy, returns in results.items():
            if returns:
                total_return = np.prod([1 + r for r in returns]) - 1
                avg_return = np.mean(returns)
                win_rate = sum(1 for r in returns if r > 0) / len(returns)
                
                print(f"\n{strategy}:")
                print(f"  Total Return: {total_return*100:+.2f}%")
                print(f"  Avg Monthly: {avg_return*100:+.2f}%")
                print(f"  Win Rate: {win_rate*100:.0f}%")
                print(f"  Trades: {len(returns)}")
        
        return results
    
    def compare_consistency(self, results):
        """Compare consistency metrics"""
        
        print(f"\n{'='*60}")
        print("CONSISTENCY ANALYSIS")
        print(f"{'='*60}")
        
        for strategy, returns in results.items():
            if returns and len(returns) > 1:
                # Calculate rolling volatility
                volatility = np.std(returns)
                
                # Calculate max drawdown
                cumulative = np.cumprod([1 + r for r in returns])
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown)
                
                # Calculate consistency score (lower volatility = higher consistency)
                consistency_score = 1 / (1 + volatility)
                
                print(f"\n{strategy}:")
                print(f"  Volatility: {volatility*100:.2f}%")
                print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
                print(f"  Consistency Score: {consistency_score:.3f}")


def main():
    print("="*80)
    print("LSTM CONSISTENCY IMPROVEMENT EXPERIMENT")
    print("Testing adaptive gamma selection for better consistency")
    print("="*80)
    
    experiment = LSTMConsistencyExperiment()
    
    # Run for 2021
    results_2021 = experiment.run_adaptive_strategy(2021)
    
    # Compare consistency
    experiment.compare_consistency(results_2021)
    
    # Save results
    output_file = f"{experiment.output_dir}/results.json"
    with open(output_file, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {k: [float(v) for v in vals] for k, vals in results_2021.items()}
        json.dump(json_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_2021


if __name__ == "__main__":
    results = main()