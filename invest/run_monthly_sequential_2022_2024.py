"""
Run Sequential Supervised Learning with Monthly Trading (12x per year)
for 2022, 2023, and 2024 to properly compare with DQN/TD3
"""

import torch
import pickle
import numpy as np
import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

from model.iimodel import IIMODEL


@dataclass
class MonthlyTradingConfig:
    """Configuration for monthly trading experiment."""
    years: List[int] = None
    gamma: float = 0.3
    learning_rate: float = 0.001
    training_steps: int = 200  # Balanced between speed and performance
    num_consecutive_days: int = 7
    obj_use_mean_return: bool = False
    device: str = 'cuda'
    seed: int = 42
    data_dir: str = '/home/ubuntu/code/angle_rl/invest/data/'
    output_dir: str = None
    
    def __post_init__(self):
        if self.years is None:
            self.years = [2022, 2023, 2024]
        if self.output_dir is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_dir = f'{self.data_dir}seq_monthly_12x_{timestamp}/'


class MonthlySequentialTrading:
    """Monthly trading strategy using sequential supervised learning."""
    
    def __init__(self, config: MonthlyTradingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load data files
        self.data_files = self._load_data_files()
        
        # Results tracking
        self.results = {
            'config': config.__dict__,
            'annual_results': {},
            'overall_performance': {}
        }
    
    def _load_data_files(self) -> List[str]:
        """Load list of available data files."""
        list_file = f'{self.config.data_dir}all_data_list.txt'
        with open(list_file, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(files)} data files")
        return files
    
    def _extract_date_from_filename(self, filename: str) -> str:
        """Extract test date from filename."""
        pattern = r'test_data_start_date_(\d{4}_\d{2}_\d{2})'
        match = re.search(pattern, filename)
        if match:
            return match.group(1).replace('_', '-')
        return None
    
    def _find_files_for_month(self, year: int, month: int) -> Tuple[List[str], str]:
        """Find data files for a specific year and month."""
        
        # Look for files with test dates in the target month
        target_patterns = [
            f"{year}_{month:02d}_",
            f"{year}-{month:02d}-"
        ]
        
        # Find all matching files
        matching_files = []
        for i, filepath in enumerate(self.data_files):
            date_str = self._extract_date_from_filename(filepath)
            if date_str:
                file_year = int(date_str[:4])
                file_month = int(date_str[5:7])
                if file_year == year and file_month == month:
                    matching_files.append((i, filepath, date_str))
        
        if not matching_files:
            return [], None
        
        # Use the first available file in that month
        file_idx, test_file, test_date = matching_files[0]
        
        # Get 7 consecutive training files
        train_files = []
        for i in range(file_idx - 6, file_idx + 1):
            if 0 <= i < len(self.data_files):
                train_files.append(self.data_files[i])
        
        if len(train_files) < 7:
            return [], None
        
        return train_files, test_file
    
    def train_model_for_month(self, 
                              train_files: List[str],
                              model_id: str) -> IIMODEL:
        """Train a model on consecutive data files."""
        
        # Load consecutive data files
        data_sequence = []
        for file_path in train_files:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data_sequence.append(pickle.load(f))
        
        if len(data_sequence) < 7:
            return None
        
        # Initialize model
        model = IIMODEL(
            dropout_ratio=0.0,
            num_conv_filters=32,
            hidden_dim=47
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        T = len(data_sequence)
        gamma = torch.tensor([self.config.gamma]).to(self.device)
        
        # Training loop
        for step in range(1, self.config.training_steps + 1):
            model.train()
            optimizer.zero_grad()
            
            total_loss = torch.tensor(0.0).to(self.device)
            
            for i in range(T):
                features = data_sequence[i]['trainFeature'].to(self.device)
                series = data_sequence[i]['train_in_portfolio_series'].to(self.device)
                
                # Forward pass
                portfolio_weights = model(features)
                
                # Calculate portfolio metrics
                portfolio_shares = portfolio_weights / (series[:, 0:1] + 1e-10)
                actual_return = torch.sum((series[:, -1:] - series[:, 0:1]) * portfolio_shares)
                
                returns_series = torch.sum(
                    series[:, 1:] * portfolio_shares - series[:, 0:1] * portfolio_shares,
                    dim=0
                )
                
                mean_return = torch.mean(returns_series)
                stddev = torch.std(returns_series) + 1e-10
                
                if self.config.obj_use_mean_return:
                    sharpe = mean_return / stddev
                else:
                    sharpe = actual_return / stddev
                
                # Discounted loss
                loss = -sharpe * torch.pow(gamma, T - i - 1)
                total_loss = total_loss + loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            # Log every 50 steps
            if step % 50 == 0:
                print(f"    Step {step}/{self.config.training_steps} | Loss: {total_loss.item():.4f}", end='\r')
        
        # Save model
        model_path = f'{self.config.output_dir}{model_id}.pt'
        torch.save(model.state_dict(), model_path)
        
        return model
    
    def evaluate_model(self, model: IIMODEL, test_file: str) -> Dict:
        """Evaluate model on test data."""
        
        if not os.path.exists(test_file):
            return {}
        
        # Load test data
        with open(test_file, 'rb') as f:
            test_data = pickle.load(f)
        
        if test_data.get('test_in_portfolio_series') is None:
            return {}
        
        model.eval()
        
        with torch.no_grad():
            features = test_data['testFeature'].to(self.device)
            series = test_data['test_in_portfolio_series'].to(self.device)
            
            # Get portfolio weights
            portfolio_weights = model(features)
            
            # Calculate metrics
            portfolio_shares = portfolio_weights / (series[:, 0:1] + 1e-10)
            actual_return = torch.sum((series[:, -1:] - series[:, 0:1]) * portfolio_shares)
            
            returns_series = torch.sum(
                series[:, 1:] * portfolio_shares - series[:, 0:1] * portfolio_shares,
                dim=0
            )
            
            mean_return = torch.mean(returns_series)
            stddev = torch.std(returns_series)
            sharpe = actual_return / (stddev + 1e-10)
            
            # Apply transaction cost (0.15%)
            actual_return = actual_return - 0.0015
        
        return {
            'actual_return': actual_return.item(),
            'mean_return': mean_return.item(),
            'stddev': stddev.item(),
            'sharpe': sharpe.item()
        }
    
    def run_year_monthly_trading(self, year: int) -> Dict:
        """Run monthly trading for a specific year (12 trades)."""
        
        print(f"\n{'='*80}")
        print(f"Running Monthly Trading (12x) for {year}")
        print(f"{'='*80}")
        
        year_results = {
            'year': year,
            'monthly_trades': [],
            'cumulative_return': 1.0
        }
        
        months_traded = 0
        
        for month in range(1, 13):
            print(f"\n{year}-{month:02d}: ", end='')
            
            # Find data files for this month
            train_files, test_file = self._find_files_for_month(year, month)
            
            if not train_files or not test_file:
                print("No data available")
                continue
            
            # Extract date for logging
            test_date = self._extract_date_from_filename(test_file)
            print(f"Trading on {test_date}")
            
            # Train model
            model_id = f"model_{year}_m{month:02d}"
            model = self.train_model_for_month(train_files, model_id)
            
            if model is None:
                print("  Training failed")
                continue
            
            # Evaluate model
            eval_results = self.evaluate_model(model, test_file)
            
            if eval_results:
                # Update cumulative return
                holding_return = eval_results['actual_return']
                year_results['cumulative_return'] *= (1 + holding_return)
                
                trade_result = {
                    'month': month,
                    'date': test_date,
                    'return': holding_return,
                    'sharpe': eval_results['sharpe'],
                    'cumulative': year_results['cumulative_return'] - 1
                }
                
                year_results['monthly_trades'].append(trade_result)
                months_traded += 1
                
                print(f"  Return: {holding_return*100:+.2f}% | Sharpe: {eval_results['sharpe']:.3f} | YTD: {(year_results['cumulative_return']-1)*100:+.2f}%")
            
            # Clear GPU memory
            del model
            torch.cuda.empty_cache()
        
        # Calculate year statistics
        if year_results['monthly_trades']:
            returns = [t['return'] for t in year_results['monthly_trades']]
            year_results['statistics'] = {
                'annual_return': year_results['cumulative_return'] - 1,
                'months_traded': months_traded,
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-10),
                'best_month': max(returns),
                'worst_month': min(returns),
                'win_rate': sum(1 for r in returns if r > 0) / len(returns)
            }
        
        return year_results
    
    def run_full_experiment(self) -> Dict:
        """Run the complete monthly trading experiment for all years."""
        
        print("\n" + "="*80)
        print("SEQUENTIAL SUPERVISED LEARNING - MONTHLY TRADING (12x per year)")
        print("="*80)
        print(f"Years: {self.config.years}")
        print(f"Training steps: {self.config.training_steps}")
        print(f"Gamma: {self.config.gamma}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Transaction cost: 0.15% per trade")
        
        overall_cumulative = 1.0
        
        for year in self.config.years:
            year_results = self.run_year_monthly_trading(year)
            self.results['annual_results'][year] = year_results
            overall_cumulative *= year_results.get('cumulative_return', 1.0)
        
        # Calculate overall performance
        all_trades = []
        for year_data in self.results['annual_results'].values():
            if 'monthly_trades' in year_data:
                all_trades.extend(year_data['monthly_trades'])
        
        if all_trades:
            all_returns = [t['return'] for t in all_trades]
            self.results['overall_performance'] = {
                'total_trades': len(all_trades),
                'avg_return_per_trade': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'sharpe_ratio': np.mean(all_returns) / (np.std(all_returns) + 1e-10),
                'cumulative_return': overall_cumulative - 1,
                'annualized_return': (overall_cumulative ** (1/len(self.config.years))) - 1,
                'best_trade': max(all_returns),
                'worst_trade': min(all_returns),
                'win_rate': sum(1 for r in all_returns if r > 0) / len(all_returns),
                'total_transaction_costs': len(all_trades) * 0.0015
            }
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results."""
        
        # Save as JSON
        json_path = f'{self.config.output_dir}results.json'
        with open(json_path, 'w') as f:
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                elif isinstance(obj, torch.device):
                    return str(obj)
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)
        
        print(f"\nResults saved to {self.config.output_dir}")
    
    def print_summary(self):
        """Print experiment summary and comparison with DQN/TD3."""
        
        print("\n" + "="*80)
        print("SUMMARY - SEQUENTIAL SUPERVISED (MONTHLY TRADING 12x/year)")
        print("="*80)
        
        # Overall performance
        if 'overall_performance' in self.results:
            perf = self.results['overall_performance']
            print(f"\nOverall Performance (2022-2024):")
            print(f"  Total trades: {perf.get('total_trades', 0)}")
            print(f"  Cumulative return: {perf.get('cumulative_return', 0)*100:+.2f}%")
            print(f"  Annualized return: {perf.get('annualized_return', 0)*100:+.2f}%")
            print(f"  Average per trade: {perf.get('avg_return_per_trade', 0)*100:+.2f}%")
            print(f"  Sharpe ratio: {perf.get('sharpe_ratio', 0):.4f}")
            print(f"  Win rate: {perf.get('win_rate', 0)*100:.1f}%")
            print(f"  Best/Worst trade: {perf.get('best_trade', 0)*100:+.2f}% / {perf.get('worst_trade', 0)*100:+.2f}%")
        
        # Annual performance
        print("\nAnnual Performance:")
        print("-" * 40)
        for year in self.config.years:
            if year in self.results.get('annual_results', {}):
                data = self.results['annual_results'][year]
                if 'statistics' in data:
                    stats = data['statistics']
                    print(f"\n{year}:")
                    print(f"  Annual return: {stats['annual_return']*100:+.2f}%")
                    print(f"  Months traded: {stats['months_traded']}/12")
                    print(f"  Avg monthly: {stats['avg_return']*100:+.2f}%")
                    print(f"  Sharpe: {stats['sharpe']:.4f}")
                    print(f"  Win rate: {stats['win_rate']*100:.1f}%")
        
        # Comparison with DQN/TD3
        print("\n" + "="*80)
        print("COMPARISON WITH DQN/TD3")
        print("="*80)
        
        print("""
        DQN/TD3 Results (from ablation study):
        ----------------------------------------
        Period labeled "2023" (actually 2022 data):
          - Evaluation window: Apr-Jun 2022
          - Average return: -8.54%
        
        Period labeled "2024" (actually 2023 data):
          - Evaluation window: Mar-May 2023  
          - Average return: -0.51%
        
        Overall DQN/TD3 average: -4.52%
        """)
        
        if 'overall_performance' in self.results:
            avg_return = self.results['overall_performance'].get('avg_return_per_trade', 0)
            print(f"\nSequential Supervised (Monthly 12x):")
            print(f"  Average return per trade: {avg_return*100:+.2f}%")
            
            improvement = avg_return - (-0.0452)
            print(f"\n  Improvement over DQN/TD3: {improvement*100:+.2f} percentage points")


def main():
    """Main entry point."""
    
    # Create configuration for monthly trading
    config = MonthlyTradingConfig(
        years=[2022, 2023, 2024],
        gamma=0.3,
        learning_rate=0.001,
        training_steps=200,  # Balanced for performance
        num_consecutive_days=7,
        obj_use_mean_return=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run experiment
    experiment = MonthlySequentialTrading(config)
    results = experiment.run_full_experiment()
    
    # Print summary
    experiment.print_summary()
    
    return results


if __name__ == "__main__":
    results = main()