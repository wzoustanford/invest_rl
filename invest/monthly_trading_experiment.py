"""
Monthly Trading Experiment for Sequential Supervised Learning

This module implements a monthly trading strategy using sequential supervised learning
for the years 2023-2024. The strategy:
- Trains on 7-day sequences to predict 25-day returns
- Executes 12 trades per year (monthly)
- Uses optimized GPU memory management
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import torch
import gc

from sequential_supervised_trainer import SequentialSupervisedTrainer, TrainingConfig


@dataclass
class TradingConfig:
    """Configuration for monthly trading experiment."""
    years: List[int] = None  # e.g., [2023, 2024]
    trades_per_year: int = 12  # Monthly trading
    sequence_days: int = 7  # Number of consecutive days for training
    holding_period_days: int = 25  # How long to hold positions
    training_window_days: int = 360  # Historical data for training
    
    # Model parameters
    gamma: float = 0.3
    learning_rate: float = 0.001
    training_steps: int = 750
    
    # Paths
    data_dir: str = '/home/ubuntu/code/angle_rl/invest/data/'
    output_dir: str = None
    
    def __post_init__(self):
        if self.years is None:
            self.years = [2023, 2024]
        if self.output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f'{self.data_dir}monthly_trading_exp_{timestamp}/'


class MonthlyTradingExperiment:
    """Main class for running monthly trading experiments."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load data file list
        self.data_files = self._load_data_file_list()
        
        # Initialize results tracking
        self.results = {
            'config': asdict(config),
            'monthly_trades': [],
            'annual_returns': {},
            'overall_performance': {}
        }
    
    def _load_data_file_list(self) -> List[str]:
        """Load the list of available data files."""
        list_file = f'{self.config.data_dir}all_data_list.txt'
        
        # If all_data_list doesn't exist, try to find another list file
        if not os.path.exists(list_file):
            # Look for data list files with date patterns
            import glob
            pattern = f'{self.config.data_dir}data_list_*.txt'
            list_files = glob.glob(pattern)
            if list_files:
                list_file = list_files[0]  # Use first available
                print(f"Using data list file: {list_file}")
            else:
                raise FileNotFoundError(f"No data list files found in {self.config.data_dir}")
        
        with open(list_file, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        return files
    
    def _find_data_files_for_date(self, target_date: str) -> Tuple[List[str], str]:
        """
        Find sequence of data files for a given target date.
        
        Returns:
            Tuple of (training_files, test_file)
        """
        # Find file that contains the target date in its test period
        test_file = None
        for file_path in self.data_files:
            if target_date in file_path:
                test_file = file_path
                break
        
        if not test_file:
            # Try to find closest file
            for file_path in self.data_files:
                # Extract date from filename pattern
                if 'test_data_start_date' in file_path:
                    parts = file_path.split('test_data_start_date_')[1]
                    file_date = parts.split('_')[0]
                    if file_date >= target_date:
                        test_file = file_path
                        break
        
        if not test_file:
            return [], None
        
        # Find index of test file
        try:
            test_idx = self.data_files.index(test_file)
        except ValueError:
            return [], None
        
        # Get sequence of training files (7 consecutive days)
        train_files = []
        for i in range(self.config.sequence_days):
            idx = test_idx - (self.config.sequence_days - i - 1)
            if idx >= 0 and idx < len(self.data_files):
                train_files.append(self.data_files[idx])
        
        if len(train_files) != self.config.sequence_days:
            return [], None
        
        return train_files, test_file
    
    def _generate_trading_dates(self, year: int) -> List[str]:
        """Generate monthly trading dates for a given year."""
        dates = []
        
        for month in range(1, 13):
            # Trade on first trading day of each month
            # For simplicity, use 5th of each month (to avoid holidays)
            date = datetime(year, month, 5)
            date_str = date.strftime('%Y-%m-%d')
            dates.append(date_str)
        
        return dates
    
    def train_model_for_date(self, 
                            trading_date: str,
                            train_files: List[str],
                            model_name: str) -> Optional[SequentialSupervisedTrainer]:
        """Train model for a specific trading date."""
        
        # Create training configuration
        train_config = TrainingConfig(
            gamma=self.config.gamma,
            learning_rate=self.config.learning_rate,
            num_steps=self.config.training_steps,
            num_consecutive_days=self.config.sequence_days,
            obj_use_mean_return=False,  # Use actual return for Sharpe
            gradient_accumulation_steps=4,
            use_mixed_precision=True,
            log_interval=100,
            eval_interval=100,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize trainer
        trainer = SequentialSupervisedTrainer(train_config)
        
        # Train on sequence
        save_dir = os.path.join(self.config.output_dir, f'models_{trading_date}')
        history = trainer.train_on_sequence(
            data_files=train_files,
            save_dir=save_dir,
            model_name=model_name
        )
        
        return trainer
    
    def evaluate_trading_performance(self,
                                    trainer: SequentialSupervisedTrainer,
                                    test_file: str,
                                    trading_date: str) -> Dict:
        """Evaluate trading performance for a specific date."""
        
        # Load test data
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        # Evaluate model
        eval_results = trainer.evaluate(test_data)
        
        # Calculate annualized metrics
        holding_return = eval_results.get('actual_return', 0)
        periods_per_year = 365 / self.config.holding_period_days
        annualized_return = (1 + holding_return) ** periods_per_year - 1
        
        results = {
            'trading_date': trading_date,
            'test_file': test_file,
            'holding_period_return': holding_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': eval_results.get('sharpe', 0),
            'stddev': eval_results.get('stddev', 0),
            'mean_return': eval_results.get('mean_return', 0),
            'top_stocks': eval_results.get('top_stocks', [])[:10],  # Top 10 stocks
            'portfolio_concentration': self._calculate_concentration(
                eval_results.get('portfolio_weights', np.array([]))
            )
        }
        
        return results
    
    def _calculate_concentration(self, weights: np.ndarray) -> float:
        """Calculate portfolio concentration (Herfindahl index)."""
        if len(weights) == 0:
            return 0.0
        weights = weights.flatten()
        return float(np.sum(weights ** 2))
    
    def run_year_experiment(self, year: int) -> Dict:
        """Run monthly trading experiment for a specific year."""
        
        print(f"\n{'='*60}")
        print(f"Running Monthly Trading Experiment for {year}")
        print(f"{'='*60}")
        
        year_results = {
            'year': year,
            'monthly_trades': [],
            'cumulative_return': 1.0
        }
        
        trading_dates = self._generate_trading_dates(year)
        
        for i, trading_date in enumerate(trading_dates):
            print(f"\n--- Month {i+1}/12: Trading on {trading_date} ---")
            
            # Find appropriate data files
            train_files, test_file = self._find_data_files_for_date(trading_date)
            
            if not train_files or not test_file:
                print(f"Warning: No data available for {trading_date}, skipping...")
                continue
            
            # Train model
            model_name = f"model_{year}_month{i+1:02d}"
            trainer = self.train_model_for_date(
                trading_date=trading_date,
                train_files=train_files,
                model_name=model_name
            )
            
            # Evaluate performance
            trade_results = self.evaluate_trading_performance(
                trainer=trainer,
                test_file=test_file,
                trading_date=trading_date
            )
            
            # Update cumulative return
            year_results['cumulative_return'] *= (1 + trade_results['holding_period_return'])
            trade_results['cumulative_return'] = year_results['cumulative_return']
            
            year_results['monthly_trades'].append(trade_results)
            
            # Print summary
            print(f"  Holding Period Return: {trade_results['holding_period_return']*100:.2f}%")
            print(f"  Sharpe Ratio: {trade_results['sharpe_ratio']:.4f}")
            print(f"  Cumulative Return YTD: {(year_results['cumulative_return']-1)*100:.2f}%")
            
            # Clean up memory
            del trainer
            torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate year statistics
        returns = [t['holding_period_return'] for t in year_results['monthly_trades']]
        year_results['statistics'] = {
            'total_trades': len(year_results['monthly_trades']),
            'annual_return': year_results['cumulative_return'] - 1,
            'avg_trade_return': np.mean(returns) if returns else 0,
            'std_trade_return': np.std(returns) if returns else 0,
            'best_trade': max(returns) if returns else 0,
            'worst_trade': min(returns) if returns else 0,
            'win_rate': sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        }
        
        return year_results
    
    def run_experiment(self) -> Dict:
        """Run the complete monthly trading experiment."""
        
        print(f"\nStarting Monthly Trading Experiment")
        print(f"Years: {self.config.years}")
        print(f"Trades per year: {self.config.trades_per_year}")
        print(f"Sequence days: {self.config.sequence_days}")
        print(f"Holding period: {self.config.holding_period_days} days")
        print(f"Output directory: {self.config.output_dir}")
        
        overall_cumulative = 1.0
        
        for year in self.config.years:
            year_results = self.run_year_experiment(year)
            self.results['annual_returns'][year] = year_results
            overall_cumulative *= year_results['cumulative_return']
        
        # Calculate overall statistics
        all_trades = []
        for year_data in self.results['annual_returns'].values():
            all_trades.extend(year_data['monthly_trades'])
        
        all_returns = [t['holding_period_return'] for t in all_trades]
        
        self.results['overall_performance'] = {
            'total_trades': len(all_trades),
            'cumulative_return': overall_cumulative - 1,
            'annualized_return': (overall_cumulative ** (1/len(self.config.years))) - 1,
            'avg_trade_return': np.mean(all_returns) if all_returns else 0,
            'std_trade_return': np.std(all_returns) if all_returns else 0,
            'sharpe_ratio': np.mean(all_returns) / (np.std(all_returns) + 1e-10) if all_returns else 0,
            'best_trade': max(all_returns) if all_returns else 0,
            'worst_trade': min(all_returns) if all_returns else 0,
            'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns) if all_returns else 0,
            'max_drawdown': self._calculate_max_drawdown(all_trades)
        }
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _calculate_max_drawdown(self, trades: List[Dict]) -> float:
        """Calculate maximum drawdown from trade history."""
        if not trades:
            return 0.0
        
        cumulative = []
        current = 1.0
        for trade in trades:
            current *= (1 + trade['holding_period_return'])
            cumulative.append(current)
        
        peak = cumulative[0]
        max_dd = 0
        
        for value in cumulative:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _save_results(self):
        """Save experiment results to files."""
        
        # Save as JSON
        json_path = os.path.join(self.config.output_dir, 'results.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save as pickle for full data
        pkl_path = os.path.join(self.config.output_dir, 'results.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.results, f)
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"\nResults saved to {self.config.output_dir}")
    
    def _generate_summary_report(self):
        """Generate a human-readable summary report."""
        
        report_path = os.path.join(self.config.output_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MONTHLY TRADING EXPERIMENT - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-"*40 + "\n")
            f.write(f"Years: {self.config.years}\n")
            f.write(f"Trades per year: {self.config.trades_per_year}\n")
            f.write(f"Sequence days: {self.config.sequence_days}\n")
            f.write(f"Holding period: {self.config.holding_period_days} days\n")
            f.write(f"Training steps: {self.config.training_steps}\n")
            f.write(f"Gamma: {self.config.gamma}\n")
            f.write(f"Learning rate: {self.config.learning_rate}\n\n")
            
            # Overall Performance
            perf = self.results['overall_performance']
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            f.write(f"Total trades: {perf['total_trades']}\n")
            f.write(f"Cumulative return: {perf['cumulative_return']*100:.2f}%\n")
            f.write(f"Annualized return: {perf['annualized_return']*100:.2f}%\n")
            f.write(f"Average trade return: {perf['avg_trade_return']*100:.2f}%\n")
            f.write(f"Std dev of returns: {perf['std_trade_return']*100:.2f}%\n")
            f.write(f"Sharpe ratio: {perf['sharpe_ratio']:.4f}\n")
            f.write(f"Win rate: {perf['win_rate']*100:.1f}%\n")
            f.write(f"Best trade: {perf['best_trade']*100:.2f}%\n")
            f.write(f"Worst trade: {perf['worst_trade']*100:.2f}%\n")
            f.write(f"Max drawdown: {perf['max_drawdown']*100:.2f}%\n\n")
            
            # Annual Performance
            f.write("ANNUAL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            for year, data in self.results['annual_returns'].items():
                stats = data['statistics']
                f.write(f"\n{year}:\n")
                f.write(f"  Annual return: {stats['annual_return']*100:.2f}%\n")
                f.write(f"  Total trades: {stats['total_trades']}\n")
                f.write(f"  Average return: {stats['avg_trade_return']*100:.2f}%\n")
                f.write(f"  Win rate: {stats['win_rate']*100:.1f}%\n")
                f.write(f"  Best/Worst: {stats['best_trade']*100:.2f}% / {stats['worst_trade']*100:.2f}%\n")
            
            # Monthly Trade Details
            f.write("\n\nMONTHLY TRADE DETAILS\n")
            f.write("-"*40 + "\n")
            for year, data in self.results['annual_returns'].items():
                f.write(f"\n{year}:\n")
                for i, trade in enumerate(data['monthly_trades'], 1):
                    f.write(f"  Month {i:2d} ({trade['trading_date']}): "
                           f"{trade['holding_period_return']*100:6.2f}% "
                           f"(Sharpe: {trade['sharpe_ratio']:.3f})\n")


def main():
    """Main entry point for running the monthly trading experiment."""
    
    # Create configuration
    config = TradingConfig(
        years=[2023, 2024],
        trades_per_year=12,
        sequence_days=7,
        holding_period_days=25,
        training_window_days=360,
        gamma=0.3,
        learning_rate=0.001,
        training_steps=750
    )
    
    # Run experiment
    experiment = MonthlyTradingExperiment(config)
    results = experiment.run_experiment()
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("="*80)
    
    perf = results['overall_performance']
    print(f"\nOverall Performance (2023-2024):")
    print(f"  Cumulative Return: {perf['cumulative_return']*100:.2f}%")
    print(f"  Annualized Return: {perf['annualized_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {perf['sharpe_ratio']:.4f}")
    print(f"  Win Rate: {perf['win_rate']*100:.1f}%")
    print(f"  Max Drawdown: {perf['max_drawdown']*100:.2f}%")
    
    print("\nAnnual Returns:")
    for year, data in results['annual_returns'].items():
        print(f"  {year}: {data['statistics']['annual_return']*100:.2f}%")
    
    print(f"\nResults saved to: {config.output_dir}")


if __name__ == "__main__":
    main()