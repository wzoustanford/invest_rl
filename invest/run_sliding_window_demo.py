#!/usr/bin/env python3
"""
Demo script to run sliding window experiment without interactive prompts.
This runs a full 4-window experiment automatically.
"""

import sys
import os
import time
sys.path.append('/home/ubuntu/code/angle_rl/invest')

from sliding_window_experiment import SlidingWindowExperiment


def main():
    """Run sliding window experiment demo."""
    
    print("=== Sliding Window Experiment Demo ===")
    print("This will run the full 4-window experiment automatically.")
    print("Expected runtime: ~1.1 hours")
    print("Results will be saved to JSON files for analysis.")
    print()
    
    # Create experiment
    experiment = SlidingWindowExperiment(
        target_eval_days=240,  # ~1 year of evaluation
        training_window_size=265,
        eval_window_size=60,
        window_shift=60,
        base_exp_id="sliding_window_demo"
    )
    
    # Print experiment plan
    experiment.print_experiment_plan()
    
    print(f"\n=== Starting DQN Sliding Window Experiment ===")
    start_time = time.time()
    
    # Run DQN experiment
    dqn_results = experiment.run_full_experiment(algorithm='dqn')
    
    total_time = time.time() - start_time
    
    print(f"\n=== Experiment Complete ===")
    print(f"Total runtime: {total_time/3600:.2f} hours")
    
    # Print summary
    if 'summary' in dqn_results and 'evaluation_performance' in dqn_results['summary']:
        summary = dqn_results['summary']
        eval_perf = summary['evaluation_performance']
        
        print(f"\n=== DQN Results Summary ===")
        print(f"Successful windows: {summary['successful_windows']}/{summary['total_windows']}")
        print(f"Total evaluation days: {summary['total_eval_days']}")
        print(f"Average return: {eval_perf['mean_return_pct']:.2f}% ± {eval_perf['std_return_pct']:.2f}%")
        print(f"Return range: {eval_perf['min_return_pct']:.2f}% to {eval_perf['max_return_pct']:.2f}%")
        print(f"Individual returns: {[f'{r:.1f}%' for r in eval_perf['returns_list']]}")
        
        # Calculate cumulative return
        cumulative_return = 1.0
        for ret in eval_perf['returns_list']:
            cumulative_return *= (1 + ret/100)
        
        total_return_pct = (cumulative_return - 1.0) * 100
        print(f"Cumulative return over {summary['total_eval_days']} days: {total_return_pct:.2f}%")
        
        # Annualized return (assuming 252 trading days per year)
        trading_days = summary['total_eval_days']
        annualized_return = ((cumulative_return ** (252 / trading_days)) - 1) * 100
        print(f"Annualized return: {annualized_return:.2f}%")
    
    # Save compact results summary
    results_summary = {
        'experiment': 'sliding_window_demo',
        'algorithm': 'dqn',
        'runtime_hours': total_time / 3600,
        'windows_completed': dqn_results['summary']['successful_windows'] if 'summary' in dqn_results else 0,
        'total_eval_days': dqn_results['summary']['total_eval_days'] if 'summary' in dqn_results else 0,
        'performance': dqn_results['summary']['evaluation_performance'] if 'summary' in dqn_results else None
    }
    
    import json
    with open('/home/ubuntu/code/angle_rl/invest/sliding_window_demo_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nResults summary saved to: sliding_window_demo_summary.json")
    print(f"Full results saved to: sliding_window_demo_dqn_results.json")
    
    return experiment, dqn_results


if __name__ == "__main__":
    experiment, results = main()