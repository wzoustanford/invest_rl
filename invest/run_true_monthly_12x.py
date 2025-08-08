"""
True Monthly Trading (12x per year) - Sequential Supervised Learning
Exactly as specified: 12 trades per year, one for each month
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple

from model.iimodel import IIMODEL


class TrueMonthlyTrading:
    """Implement true monthly trading (12x per year) as specified."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Configuration
        self.gamma = 0.3
        self.learning_rate = 0.001
        self.training_steps = 100  # Can increase for better performance
        self.num_files_for_training = 7  # 7 consecutive files before trading day
        
        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = f'/home/ubuntu/code/angle_rl/invest/data/true_monthly_12x_{timestamp}/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load all data files
        self.data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
        with open(f'{self.data_dir}all_data_list.txt', 'r') as f:
            self.all_files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.all_files)} data files")
        
        # Results storage
        self.results = {
            'config': {
                'gamma': self.gamma,
                'learning_rate': self.learning_rate,
                'training_steps': self.training_steps,
                'trades_per_year': 12,
                'transaction_cost': 0.0015
            },
            'years': {}
        }
    
    def extract_date_from_filename(self, filename: str) -> Tuple[str, str]:
        """Extract training and test dates from filename."""
        pattern = r'training_data_start_date_(\d{4}_\d{2}_\d{2})_test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            train_date = match.group(1).replace('_', '-')
            test_date = match.group(2).replace('_', '-')
            return train_date, test_date
        return None, None
    
    def find_file_for_month(self, year: int, month: int) -> Tuple[int, str]:
        """Find the file index for a specific year and month."""
        # Look for files with test date in the target month
        for i, filepath in enumerate(self.all_files):
            _, test_date = self.extract_date_from_filename(filepath)
            if test_date:
                file_year = int(test_date[:4])
                file_month = int(test_date[5:7])
                # Find first file in that month
                if file_year == year and file_month == month:
                    return i, filepath
        return None, None
    
    def train_model_for_month(self, training_files: List[str]) -> torch.nn.Module:
        """
        Train model on 7 consecutive files using gamma-discounted Sharpe losses.
        Each file represents one day's trading with ~25 day holding period.
        """
        
        # Load the 7 consecutive data files
        data_sequence = []
        for filepath in training_files:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data_sequence.append(pickle.load(f))
            else:
                print(f"Warning: File not found: {filepath}")
                return None
        
        if len(data_sequence) != 7:
            print(f"Error: Expected 7 files, got {len(data_sequence)}")
            return None
        
        # Initialize model
        model = IIMODEL(
            dropout_ratio=0.0,
            num_conv_filters=32,
            hidden_dim=47
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Training loop
        for step in range(1, self.training_steps + 1):
            model.train()
            optimizer.zero_grad()
            
            # Aggregate loss across all 7 files with gamma discounting
            total_loss = torch.tensor(0.0).to(self.device)
            
            for i in range(7):
                # Get features and series for this file
                features = data_sequence[i]['trainFeature'].to(self.device)
                series = data_sequence[i]['train_in_portfolio_series'].to(self.device)
                
                # Forward pass: features → model → portfolio weights
                portfolio_weights = model(features)
                
                # Calculate portfolio return
                # portfolio_weights: [num_stocks, 1] - softmax output
                # series: [num_stocks, ~25 days] - price series
                
                # Buy at day 1 price, sell at last day price
                initial_prices = series[:, 0:1] + 1e-10  # Avoid division by zero
                final_prices = series[:, -1:]
                
                # Shares bought with portfolio allocation
                portfolio_shares = portfolio_weights / initial_prices
                
                # Return for each stock
                stock_returns = (final_prices - series[:, 0:1]) * portfolio_shares
                portfolio_return = torch.sum(stock_returns)
                
                # Calculate daily returns for Sharpe ratio
                daily_portfolio_values = torch.sum(series * portfolio_shares, dim=0)
                daily_returns = daily_portfolio_values[1:] - daily_portfolio_values[:-1]
                
                # Sharpe ratio = mean_return / std_dev
                mean_daily_return = torch.mean(daily_returns)
                std_daily_return = torch.std(daily_returns) + 1e-10
                sharpe_ratio = portfolio_return / std_daily_return
                
                # Apply gamma discounting (most recent file has highest weight)
                # File 0 is 7 days before trading, file 6 is 1 day before
                gamma_power = self.gamma ** (7 - i - 1)
                
                # Loss = -Sharpe (we maximize Sharpe by minimizing negative Sharpe)
                loss = -sharpe_ratio * gamma_power
                total_loss = total_loss + loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Print progress
            if step % 25 == 0:
                print(f"    Training step {step}/{self.training_steps}, Loss: {total_loss.item():.4f}", end='\r')
        
        print()  # New line after training
        return model
    
    def evaluate_monthly_trade(self, model: torch.nn.Module, test_file: str) -> Dict:
        """
        Evaluate the model on the actual trading month.
        Returns the portfolio return for the 25-day holding period.
        """
        
        if not os.path.exists(test_file):
            return None
        
        # Load test data
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        # Check if test data exists
        if test_data.get('test_in_portfolio_series') is None:
            return None
        
        model.eval()
        
        with torch.no_grad():
            # Get features and series for evaluation
            features = test_data['testFeature'].to(self.device)
            series = test_data['test_in_portfolio_series'].to(self.device)
            
            # Get portfolio weights from trained model
            portfolio_weights = model(features)
            
            # Calculate actual return
            initial_prices = series[:, 0:1] + 1e-10
            final_prices = series[:, -1:]
            
            portfolio_shares = portfolio_weights / initial_prices
            stock_returns = (final_prices - series[:, 0:1]) * portfolio_shares
            portfolio_return = torch.sum(stock_returns)
            
            # Calculate Sharpe for reporting
            daily_portfolio_values = torch.sum(series * portfolio_shares, dim=0)
            daily_returns = daily_portfolio_values[1:] - daily_portfolio_values[:-1]
            std_daily_return = torch.std(daily_returns) + 1e-10
            sharpe_ratio = portfolio_return / std_daily_return
            
            # Apply transaction cost (0.15%)
            portfolio_return = portfolio_return - 0.0015
            
            return {
                'return': portfolio_return.item(),
                'sharpe': sharpe_ratio.item(),
                'std': std_daily_return.item()
            }
    
    def run_monthly_trading_for_year(self, year: int) -> Dict:
        """
        Run 12 monthly trades for a given year.
        Each month: train on 7 preceding files, trade on month's file.
        """
        
        print(f"\n{'='*80}")
        print(f"Running TRUE MONTHLY TRADING (12x) for {year}")
        print(f"{'='*80}")
        
        year_results = {
            'year': year,
            'monthly_trades': [],
            'cumulative_return': 1.0
        }
        
        successful_trades = 0
        
        # Trade every month (12 times)
        for month in range(1, 13):
            print(f"\nMonth {month:02d}/{year}:")
            
            # Find the file for this month
            trade_file_idx, trade_file = self.find_file_for_month(year, month)
            
            if trade_file_idx is None:
                print(f"  No data available for {year}-{month:02d}")
                continue
            
            # Get 7 training files (the 7 files before the trading file)
            if trade_file_idx < 6:
                print(f"  Insufficient training data (need 7 prior files)")
                continue
            
            training_files = []
            for i in range(trade_file_idx - 6, trade_file_idx + 1):
                training_files.append(self.all_files[i])
            
            # Show what files we're using
            _, test_date = self.extract_date_from_filename(trade_file)
            print(f"  Trading date: {test_date}")
            print(f"  Training on files {trade_file_idx-6} to {trade_file_idx}")
            
            # Train model on 7 consecutive files
            print(f"  Training model on 7 consecutive days...")
            model = self.train_model_for_month(training_files)
            
            if model is None:
                print(f"  Training failed")
                continue
            
            # Evaluate on the trading month
            print(f"  Evaluating trade...")
            trade_result = self.evaluate_monthly_trade(model, trade_file)
            
            if trade_result is None:
                print(f"  Evaluation failed")
                continue
            
            # Record results
            monthly_return = trade_result['return']
            year_results['cumulative_return'] *= (1 + monthly_return)
            
            trade_record = {
                'month': month,
                'date': test_date,
                'file_idx': trade_file_idx,
                'return': monthly_return,
                'sharpe': trade_result['sharpe'],
                'cumulative_ytd': year_results['cumulative_return'] - 1
            }
            
            year_results['monthly_trades'].append(trade_record)
            successful_trades += 1
            
            # Print results
            print(f"  Return: {monthly_return*100:+.2f}%")
            print(f"  Sharpe: {trade_result['sharpe']:.3f}")
            print(f"  YTD: {(year_results['cumulative_return']-1)*100:+.2f}%")
            
            # Clean up memory
            del model
            torch.cuda.empty_cache()
        
        # Calculate year statistics
        if year_results['monthly_trades']:
            returns = [t['return'] for t in year_results['monthly_trades']]
            year_results['statistics'] = {
                'annual_return': year_results['cumulative_return'] - 1,
                'trades_executed': successful_trades,
                'trades_planned': 12,
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-10),
                'best_month': max(returns),
                'worst_month': min(returns),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns)
            }
        
        return year_results
    
    def run_experiment(self):
        """Run the complete monthly trading experiment for 2022 and 2023."""
        
        print("\n" + "="*80)
        print("TRUE MONTHLY TRADING EXPERIMENT (12x per year)")
        print("Sequential Supervised Learning with Gamma-Discounted Sharpe")
        print("="*80)
        print(f"Configuration:")
        print(f"  Gamma: {self.gamma}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Training steps: {self.training_steps}")
        print(f"  Files for training: 7 consecutive days")
        print(f"  Transaction cost: 0.15% per trade")
        
        # Run for 2022 and 2023
        for year in [2022, 2023]:
            year_results = self.run_monthly_trading_for_year(year)
            self.results['years'][year] = year_results
        
        # Calculate overall statistics
        self.calculate_overall_statistics()
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
        
        return self.results
    
    def calculate_overall_statistics(self):
        """Calculate overall performance statistics."""
        
        all_trades = []
        overall_cumulative = 1.0
        
        for year_data in self.results['years'].values():
            if 'monthly_trades' in year_data:
                all_trades.extend(year_data['monthly_trades'])
                overall_cumulative *= year_data.get('cumulative_return', 1.0)
        
        if all_trades:
            all_returns = [t['return'] for t in all_trades]
            self.results['overall'] = {
                'total_trades': len(all_trades),
                'cumulative_return': overall_cumulative - 1,
                'avg_return_per_trade': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'sharpe': np.mean(all_returns) / (np.std(all_returns) + 1e-10),
                'best_trade': max(all_returns),
                'worst_trade': min(all_returns),
                'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns)
            }
    
    def save_results(self):
        """Save results to JSON file."""
        
        json_path = os.path.join(self.output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
        
        print(f"\nResults saved to: {self.output_dir}")
    
    def print_summary(self):
        """Print comprehensive summary and comparison with DQN/TD3."""
        
        print("\n" + "="*80)
        print("SUMMARY - TRUE MONTHLY TRADING (12x per year)")
        print("="*80)
        
        # Year by year results
        for year in [2022, 2023]:
            if year in self.results['years']:
                data = self.results['years'][year]
                if 'statistics' in data:
                    stats = data['statistics']
                    print(f"\n{year}:")
                    print(f"  Annual return: {stats['annual_return']*100:+.2f}%")
                    print(f"  Trades executed: {stats['trades_executed']}/{stats['trades_planned']}")
                    print(f"  Average per trade: {stats['avg_return']*100:+.2f}%")
                    print(f"  Sharpe ratio: {stats['sharpe']:.3f}")
                    print(f"  Win rate: {stats['win_rate']*100:.0f}%")
                    print(f"  Best/Worst month: {stats['best_month']*100:+.2f}% / {stats['worst_month']*100:+.2f}%")
        
        # Overall results
        if 'overall' in self.results:
            overall = self.results['overall']
            print(f"\nOverall (2022-2023):")
            print(f"  Total trades: {overall['total_trades']}")
            print(f"  Cumulative return: {overall['cumulative_return']*100:+.2f}%")
            print(f"  Average per trade: {overall['avg_return_per_trade']*100:+.2f}%")
            print(f"  Sharpe ratio: {overall['sharpe']:.3f}")
            print(f"  Win rate: {overall['win_rate']*100:.0f}%")
        
        # Comparison with DQN/TD3
        print("\n" + "="*80)
        print("COMPARISON WITH DQN/TD3")
        print("="*80)
        print("\nDQN/TD3 Results (from ablation study):")
        print("  Average return: -4.52%")
        print("  Period tested: 2022-2023 windows")
        
        if 'overall' in self.results:
            seq_avg = self.results['overall']['avg_return_per_trade']
            print(f"\nSequential Supervised (True Monthly 12x):")
            print(f"  Average per trade: {seq_avg*100:+.2f}%")
            
            improvement = seq_avg - (-0.0452)
            print(f"\nImprovement over DQN/TD3: {improvement*100:+.2f} percentage points")


def main():
    """Run the true monthly trading experiment."""
    
    experiment = TrueMonthlyTrading()
    results = experiment.run_experiment()
    
    return results


if __name__ == "__main__":
    results = main()