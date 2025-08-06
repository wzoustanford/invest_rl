#!/usr/bin/env python3
"""
Display summary of all sliding window experiments for easy comparison.
"""

import json
import os
from datetime import datetime

def load_results(filename):
    """Load results from JSON file."""
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return None

def print_experiment_summary(name, results):
    """Print summary for one experiment."""
    if not results:
        print(f"\n{name}: Not found")
        return
    
    summary = results['summary']
    eval_perf = summary['evaluation_performance']
    train_perf = summary['training_performance']
    
    print(f"\n=== {name} ===")
    print(f"Average evaluation return: {eval_perf['mean_return_pct']:.2f}% ± {eval_perf['std_return_pct']:.2f}%")
    print(f"Average training return: {train_perf['mean_return_pct']:.2f}% ± {train_perf['std_return_pct']:.2f}%")
    print(f"Evaluation range: {eval_perf['min_return_pct']:.2f}% to {eval_perf['max_return_pct']:.2f}%")
    print(f"Window returns: {[f'{r:.1f}%' for r in eval_perf['returns_list']]}")
    print(f"Runtime: {summary['experiment_time_hours']:.2f} hours ({summary['avg_time_per_window']:.1f}s per window)")

def main():
    """Display all experiment results."""
    print(f"=== Sliding Window Experiment Summary ===")
    print(f"Generated at: {datetime.now()}")
    
    # Load all results
    dqn_5ep = load_results('/home/ubuntu/code/angle_rl/invest/sliding_window_bg_dqn_results.json')
    dqn_10ep = load_results('/home/ubuntu/code/angle_rl/invest/sliding_window_10ep_dqn_results.json')
    td3_5ep = load_results('/home/ubuntu/code/angle_rl/invest/sliding_window_td3_5ep_td3_results.json')
    
    # Print summaries
    print_experiment_summary("DQN 5 Episodes (Baseline)", dqn_5ep)
    print_experiment_summary("DQN 10 Episodes", dqn_10ep)
    print_experiment_summary("TD3 5 Episodes", td3_5ep)
    
    # Comparison table
    print("\n=== Comparison Table ===")
    print(f"{'Algorithm':<15} {'Episodes':<10} {'Eval Mean':<12} {'Eval Std':<10} {'Train Mean':<12} {'Runtime':<10}")
    print("-" * 70)
    
    if dqn_5ep:
        s = dqn_5ep['summary']
        print(f"{'DQN':<15} {'5':<10} {s['evaluation_performance']['mean_return_pct']:>11.2f}% {s['evaluation_performance']['std_return_pct']:>9.2f}% {s['training_performance']['mean_return_pct']:>11.2f}% {s['experiment_time_hours']:>9.2f}h")
    
    if dqn_10ep:
        s = dqn_10ep['summary']
        print(f"{'DQN':<15} {'10':<10} {s['evaluation_performance']['mean_return_pct']:>11.2f}% {s['evaluation_performance']['std_return_pct']:>9.2f}% {s['training_performance']['mean_return_pct']:>11.2f}% {s['experiment_time_hours']:>9.2f}h")
    
    if td3_5ep:
        s = td3_5ep['summary']
        print(f"{'TD3':<15} {'5':<10} {s['evaluation_performance']['mean_return_pct']:>11.2f}% {s['evaluation_performance']['std_return_pct']:>9.2f}% {s['training_performance']['mean_return_pct']:>11.2f}% {s['experiment_time_hours']:>9.2f}h")
    
    # Best/worst analysis
    print("\n=== Performance Analysis ===")
    all_results = [('DQN 5ep', dqn_5ep), ('DQN 10ep', dqn_10ep), ('TD3 5ep', td3_5ep)]
    valid_results = [(name, r) for name, r in all_results if r is not None]
    
    if valid_results:
        # Best evaluation performance
        best_eval = min(valid_results, key=lambda x: x[1]['summary']['evaluation_performance']['mean_return_pct'])
        print(f"Best evaluation: {best_eval[0]} ({best_eval[1]['summary']['evaluation_performance']['mean_return_pct']:.2f}%)")
        
        # Most stable (lowest std)
        most_stable = min(valid_results, key=lambda x: x[1]['summary']['evaluation_performance']['std_return_pct'])
        print(f"Most stable: {most_stable[0]} ({most_stable[1]['summary']['evaluation_performance']['std_return_pct']:.2f}% std)")
        
        # Best training
        best_train = max(valid_results, key=lambda x: x[1]['summary']['training_performance']['mean_return_pct'])
        print(f"Best training: {best_train[0]} ({best_train[1]['summary']['training_performance']['mean_return_pct']:.2f}%)")

if __name__ == "__main__":
    main()