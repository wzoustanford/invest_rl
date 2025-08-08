"""
Optimized sequential supervised learning experiment
Replicates the approach from run_sequential_action_exp_train_test.py
with performance improvements and structured reporting
"""

import torch
import pickle
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass

from model.iimodel import IIMODEL


@dataclass 
class ExperimentConfig:
    """Configuration for sequential supervised experiment."""
    exp_id: str = 'seq_supervised_monthly_2023_2024'
    gamma: float = 0.3
    learning_rate: float = 0.001
    training_steps: int = 100  # Reduced for faster testing
    num_consecutive_days: int = 7
    obj_use_mean_return: bool = False
    device: str = 'cuda'
    seed: int = 42
    
    # Data configuration
    data_dir: str = '/home/ubuntu/code/angle_rl/invest/data/'
    output_dir: str = None
    
    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = f'{self.data_dir}{self.exp_id}/'


class OptimizedSequentialExperiment:
    """Optimized version of sequential supervised learning experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load data file list
        self.data_files = self._load_data_files()
        
        # Results tracking
        self.results = {
            'config': config.__dict__,
            'training_results': [],
            'evaluation_results': []
        }
    
    def _load_data_files(self) -> List[str]:
        """Load list of available data files."""
        # Try to load from all_data_list.txt
        list_file = f'{self.config.data_dir}all_data_list.txt'
        
        if not os.path.exists(list_file):
            # Use one of the date-specific lists
            list_file = f'{self.config.data_dir}data_list_2023-04-05_2025-04-04_tr360d_bs25d_32dinterval_newsFeatureFalse_testmodeFalse.txt'
        
        with open(list_file, 'r') as f:
            files = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(files)} data files")
        return files
    
    def train_single_model(self, 
                          train_files: List[str],
                          model_id: str) -> Tuple[IIMODEL, Dict]:
        """Train a single model on consecutive data files."""
        
        print(f"\nTraining model: {model_id}")
        
        # Load consecutive data files
        data_sequence = []
        for file_path in train_files[:self.config.num_consecutive_days]:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data_sequence.append(pickle.load(f))
        
        if len(data_sequence) < 2:
            print(f"Insufficient data files for training (found {len(data_sequence)})")
            return None, {}
        
        # Initialize model
        model = IIMODEL(
            dropout_ratio=0.0,
            num_conv_filters=32,
            hidden_dim=47
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        
        T = len(data_sequence)
        gamma = torch.tensor([self.config.gamma]).to(self.device)
        
        # Training metrics
        training_metrics = {
            'losses': [],
            'returns': [],
            'sharpes': []
        }
        
        # Training loop (reduced iterations for speed)
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
            
            # Log metrics
            if step % 20 == 0:
                training_metrics['losses'].append(total_loss.item())
                training_metrics['returns'].append(actual_return.item())
                training_metrics['sharpes'].append(sharpe.item())
                
                print(f"  Step {step}/{self.config.training_steps} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Return: {actual_return.item():.4f} | "
                      f"Sharpe: {sharpe.item():.4f}")
        
        # Save model
        model_path = f'{self.config.output_dir}{model_id}.pt'
        torch.save(model.state_dict(), model_path)
        
        return model, training_metrics
    
    def evaluate_model(self, 
                      model: IIMODEL,
                      test_file: str) -> Dict:
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
            
            # Get top stocks
            top_k = min(10, len(portfolio_weights))
            top_weights, top_indices = torch.topk(portfolio_weights.squeeze(), top_k)
            
            top_stocks = []
            if 'all_test_tickers' in test_data:
                for idx in top_indices:
                    top_stocks.append(test_data['all_test_tickers'][idx])
        
        return {
            'actual_return': actual_return.item(),
            'mean_return': mean_return.item(),
            'stddev': stddev.item(),
            'sharpe': sharpe.item(),
            'top_stocks': top_stocks
        }
    
    def run_monthly_experiment(self, year: int, months: List[int] = None) -> Dict:
        """Run experiment for specific months of a year."""
        
        if months is None:
            months = [1, 4, 7, 10]  # Quarterly for faster testing
        
        print(f"\n{'='*60}")
        print(f"Running Sequential Supervised Experiment for {year}")
        print(f"Months: {months}")
        print(f"{'='*60}")
        
        year_results = {
            'year': year,
            'trades': [],
            'cumulative_return': 1.0
        }
        
        for month in months:
            print(f"\n--- Month {month}/{year} ---")
            
            # Find appropriate data files for this month
            # Look for files with test dates around this time
            target_date = f"{year}_{month:02d}"
            
            matching_files = [f for f in self.data_files if target_date in f]
            
            if not matching_files:
                # Try alternate date format
                target_date = f"{year}-{month:02d}"
                matching_files = [f for f in self.data_files if target_date in f]
            
            if not matching_files:
                print(f"No data available for {year}-{month:02d}")
                continue
            
            test_file = matching_files[0]
            test_idx = self.data_files.index(test_file)
            
            # Get training files (7 consecutive days before test)
            train_files = []
            for i in range(self.config.num_consecutive_days):
                idx = test_idx - (self.config.num_consecutive_days - i - 1)
                if 0 <= idx < len(self.data_files):
                    train_files.append(self.data_files[idx])
            
            if len(train_files) < 2:
                print(f"Insufficient training data for {year}-{month:02d}")
                continue
            
            # Train model
            model_id = f"model_{year}_m{month:02d}"
            model, train_metrics = self.train_single_model(train_files, model_id)
            
            if model is None:
                continue
            
            # Evaluate model
            eval_results = self.evaluate_model(model, test_file)
            
            if eval_results:
                # Update cumulative return
                holding_return = eval_results['actual_return']
                year_results['cumulative_return'] *= (1 + holding_return)
                
                trade_result = {
                    'month': month,
                    'return': holding_return,
                    'sharpe': eval_results['sharpe'],
                    'cumulative': year_results['cumulative_return'] - 1,
                    'top_stocks': eval_results.get('top_stocks', [])[:5]
                }
                
                year_results['trades'].append(trade_result)
                
                print(f"  Return: {holding_return*100:.2f}%")
                print(f"  Sharpe: {eval_results['sharpe']:.4f}")
                print(f"  Cumulative YTD: {(year_results['cumulative_return']-1)*100:.2f}%")
        
        # Calculate year statistics
        if year_results['trades']:
            returns = [t['return'] for t in year_results['trades']]
            year_results['statistics'] = {
                'annual_return': year_results['cumulative_return'] - 1,
                'avg_return': np.mean(returns),
                'std_return': np.std(returns),
                'sharpe': np.mean(returns) / (np.std(returns) + 1e-10),
                'num_trades': len(returns)
            }
        
        return year_results
    
    def run_full_experiment(self) -> Dict:
        """Run the complete experiment for 2023-2024."""
        
        print("\nStarting Optimized Sequential Supervised Learning Experiment")
        print(f"Configuration:")
        print(f"  Gamma: {self.config.gamma}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Training steps: {self.config.training_steps}")
        print(f"  Consecutive days: {self.config.num_consecutive_days}")
        
        # Run for 2023 and 2024
        results_2023 = self.run_monthly_experiment(2023, months=[1, 4, 7, 10])
        results_2024 = self.run_monthly_experiment(2024, months=[1, 4, 7, 10])
        
        # Combine results
        self.results['annual_results'] = {
            2023: results_2023,
            2024: results_2024
        }
        
        # Calculate overall performance
        all_returns = []
        for year_data in [results_2023, results_2024]:
            if 'trades' in year_data:
                all_returns.extend([t['return'] for t in year_data['trades']])
        
        if all_returns:
            self.results['overall_performance'] = {
                'total_trades': len(all_returns),
                'avg_return': np.mean(all_returns),
                'std_return': np.std(all_returns),
                'sharpe_ratio': np.mean(all_returns) / (np.std(all_returns) + 1e-10),
                'cumulative_2year': (results_2023.get('cumulative_return', 1.0) * 
                                    results_2024.get('cumulative_return', 1.0)) - 1
            }
        
        # Save results
        self._save_results()
        
        return self.results
    
    def _save_results(self):
        """Save experiment results."""
        
        # Save as JSON
        json_path = f'{self.config.output_dir}results.json'
        with open(json_path, 'w') as f:
            # Convert numpy types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj
            
            json.dump(self.results, f, indent=2, default=convert)
        
        print(f"\nResults saved to {self.config.output_dir}")
    
    def print_summary(self):
        """Print experiment summary."""
        
        print("\n" + "="*80)
        print("SEQUENTIAL SUPERVISED LEARNING - EXPERIMENT SUMMARY")
        print("="*80)
        
        if 'overall_performance' in self.results:
            perf = self.results['overall_performance']
            print(f"\nOverall Performance:")
            print(f"  Total trades: {perf.get('total_trades', 0)}")
            print(f"  Average return: {perf.get('avg_return', 0)*100:.2f}%")
            print(f"  Sharpe ratio: {perf.get('sharpe_ratio', 0):.4f}")
            print(f"  2-year cumulative: {perf.get('cumulative_2year', 0)*100:.2f}%")
        
        print("\nAnnual Performance:")
        for year in [2023, 2024]:
            if year in self.results.get('annual_results', {}):
                data = self.results['annual_results'][year]
                if 'statistics' in data:
                    stats = data['statistics']
                    print(f"\n{year}:")
                    print(f"  Annual return: {stats['annual_return']*100:.2f}%")
                    print(f"  Trades: {stats['num_trades']}")
                    print(f"  Sharpe: {stats['sharpe']:.4f}")
        
        print("\n" + "="*80)
        print("COMPARISON WITH DQN/TD3 RESULTS")
        print("="*80)
        print("\nDQN Results (from ablation study):")
        print("  2023: -8.54% average")
        print("  2024: -0.51% average")
        print("\nTD3 Results (from ablation study):")
        print("  2023: -8.54% average")
        print("  2024: -0.51% average")
        
        if 'overall_performance' in self.results:
            avg_ret = self.results['overall_performance'].get('avg_return', 0)
            print(f"\nSequential Supervised (this experiment):")
            print(f"  Average return: {avg_ret*100:.2f}%")
            
            improvement_dqn = avg_ret - (-0.045)  # DQN averaged -4.5%
            print(f"\n  Improvement over DQN: {improvement_dqn*100:.2f} percentage points")


def main():
    """Main entry point."""
    
    # Create configuration
    config = ExperimentConfig(
        exp_id='seq_supervised_optimized',
        gamma=0.3,
        learning_rate=0.001,
        training_steps=100,  # Reduced for faster execution
        num_consecutive_days=7,
        obj_use_mean_return=False,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run experiment
    experiment = OptimizedSequentialExperiment(config)
    results = experiment.run_full_experiment()
    
    # Print summary
    experiment.print_summary()
    
    return results


if __name__ == "__main__":
    results = main()