"""
Test script for monthly trading experiment with sequential supervised learning
"""

import os
import sys
import torch
import pickle
import numpy as np
from datetime import datetime

from monthly_trading_experiment import MonthlyTradingExperiment, TradingConfig


def test_quick_experiment():
    """Run a quick test with reduced parameters."""
    
    print("Running quick test of monthly trading experiment...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Create test configuration with reduced parameters
    config = TradingConfig(
        years=[2023],  # Just test 2023 first
        trades_per_year=3,  # Test with 3 trades instead of 12
        sequence_days=7,
        holding_period_days=25,
        training_window_days=360,
        gamma=0.3,
        learning_rate=0.001,
        training_steps=50,  # Reduced from 750 for testing
    )
    
    # Override output directory for test
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config.output_dir = f'/home/ubuntu/code/angle_rl/invest/data/test_monthly_exp_{timestamp}/'
    
    # Run experiment
    experiment = MonthlyTradingExperiment(config)
    
    # Test finding data files for specific dates
    print("\nTesting data file finding...")
    test_dates = ['2023-01-05', '2023-04-05', '2023-07-05']
    
    for date in test_dates:
        train_files, test_file = experiment._find_data_files_for_date(date)
        if train_files and test_file:
            print(f"  {date}: Found {len(train_files)} training files")
            print(f"    Test file: {os.path.basename(test_file)}")
        else:
            print(f"  {date}: No data available")
    
    # Run the experiment
    print("\nRunning test experiment...")
    results = experiment.run_experiment()
    
    # Print results
    if 'overall_performance' in results:
        perf = results['overall_performance']
        print("\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        print(f"Total trades executed: {perf.get('total_trades', 0)}")
        print(f"Average return per trade: {perf.get('avg_trade_return', 0)*100:.2f}%")
        print(f"Win rate: {perf.get('win_rate', 0)*100:.1f}%")
        
        if 2023 in results.get('annual_returns', {}):
            year_data = results['annual_returns'][2023]
            print(f"\n2023 Performance:")
            print(f"  Trades completed: {len(year_data.get('monthly_trades', []))}")
            if year_data.get('statistics'):
                stats = year_data['statistics']
                print(f"  Annual return: {stats.get('annual_return', 0)*100:.2f}%")
                print(f"  Sharpe ratio: {stats.get('avg_trade_return', 0) / (stats.get('std_trade_return', 1e-10) + 1e-10):.3f}")
    
    print(f"\nTest complete! Results saved to: {config.output_dir}")
    
    return results


def compare_with_dqn_results():
    """Compare sequential supervised results with DQN/TD3 results."""
    
    print("\n" + "="*60)
    print("COMPARISON WITH DQN/TD3 RESULTS")
    print("="*60)
    
    # Load previous DQN results if available
    dqn_results_file = '/home/ubuntu/code/angle_rl/invest/ablation_grid_search_results_20250806_224345.json'
    td3_results_file = '/home/ubuntu/code/angle_rl/invest/ablation_grid_search_td3_2ep_results_20250806_224758.json'
    
    import json
    
    results_comparison = {}
    
    # Load DQN results
    if os.path.exists(dqn_results_file):
        with open(dqn_results_file, 'r') as f:
            dqn_data = json.load(f)
            if 'summary' in dqn_data and 'year_stats' in dqn_data['summary']:
                if '2023' in dqn_data['summary']['year_stats']:
                    results_comparison['DQN_2023'] = dqn_data['summary']['year_stats']['2023']['avg_return']
                if '2024' in dqn_data['summary']['year_stats']:
                    results_comparison['DQN_2024'] = dqn_data['summary']['year_stats']['2024']['avg_return']
    
    # Load TD3 results
    if os.path.exists(td3_results_file):
        with open(td3_results_file, 'r') as f:
            td3_data = json.load(f)
            if 'summary' in td3_data and 'year_stats' in td3_data['summary']:
                if '2023' in td3_data['summary']['year_stats']:
                    results_comparison['TD3_2023'] = td3_data['summary']['year_stats']['2023']['avg_return']
                if '2024' in td3_data['summary']['year_stats']:
                    results_comparison['TD3_2024'] = td3_data['summary']['year_stats']['2024']['avg_return']
    
    print("\nPrevious Results (from ablation experiments):")
    print("-" * 40)
    for key, value in results_comparison.items():
        print(f"{key}: {value*100:.2f}%")
    
    print("\nNote: Sequential supervised learning typically performs better")
    print("because it directly optimizes the Sharpe ratio objective")
    print("using discounted rewards across consecutive trading days.")


if __name__ == "__main__":
    # Run quick test
    results = test_quick_experiment()
    
    # Compare with previous results
    compare_with_dqn_results()
    
    print("\n" + "="*60)
    print("Test script complete!")
    print("To run full experiment with all parameters, use:")
    print("  python monthly_trading_experiment.py")
    print("="*60)