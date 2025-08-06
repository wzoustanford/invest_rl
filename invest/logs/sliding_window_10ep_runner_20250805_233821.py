
import sys
import os
import time
from datetime import datetime

sys.path.append('/home/ubuntu/code/angle_rl/invest')
from sliding_window_experiment import SlidingWindowExperiment

def main():
    print(f"=== 10-Episode Sliding Window Experiment Started ===")
    print(f"Timestamp: {datetime.now()}")
    print(f"Process ID: {os.getpid()}")
    print(f"Comparison experiment: 10 episodes vs 5 episodes baseline")
    
    # Create experiment with 10 episodes per window
    experiment = SlidingWindowExperiment(
        target_eval_days=240,
        training_window_size=265,
        eval_window_size=60,
        window_shift=60,
        base_exp_id="sliding_window_10ep"
    )
    
    # Print plan
    experiment.print_experiment_plan()
    
    print(f"\n=== Starting 10-Episode DQN Experiment ===")
    print(f"Expected improvements over 5-episode baseline:")
    print(f"  - Better model convergence")
    print(f"  - More stable evaluation performance")
    print(f"  - Reduced variance across windows")
    start_time = time.time()
    
    try:
        # Run the experiment
        results = experiment.run_full_experiment(algorithm='dqn')
        
        total_time = time.time() - start_time
        print(f"\n=== 10-Episode Experiment Complete ===")
        print(f"Total runtime: {total_time/3600:.2f} hours")
        
        # Print results summary
        if 'summary' in results and 'evaluation_performance' in results['summary']:
            summary = results['summary']
            eval_perf = summary['evaluation_performance']
            train_perf = summary['training_performance']
            
            print(f"\n=== 10-Episode Results Summary ===")
            print(f"Successful windows: {summary['successful_windows']}/{summary['total_windows']}")
            print(f"Total evaluation days: {summary['total_eval_days']}")
            print(f"Average evaluation return: {eval_perf['mean_return_pct']:.2f}% ± {eval_perf['std_return_pct']:.2f}%")
            print(f"Evaluation range: {eval_perf['min_return_pct']:.2f}% to {eval_perf['max_return_pct']:.2f}%")
            print(f"Average training return: {train_perf['mean_return_pct']:.2f}% ± {train_perf['std_return_pct']:.2f}%")
            print(f"Individual evaluation returns: {[f'{r:.1f}%' for r in eval_perf['returns_list']]}")
            print(f"Individual training returns: {[f'{r:.1f}%' for r in train_perf['returns_list']]}")
            
            # Comparison with 5-episode baseline
            print(f"\n=== Comparison with 5-Episode Baseline ===")
            baseline_eval_mean = -12.56
            baseline_eval_std = 12.94
            baseline_train_mean = -7.23
            
            eval_improvement = eval_perf['mean_return_pct'] - baseline_eval_mean
            train_improvement = train_perf['mean_return_pct'] - baseline_train_mean
            
            print(f"Evaluation improvement: {eval_improvement:+.2f}% ({eval_perf['mean_return_pct']:.2f}% vs {baseline_eval_mean:.2f}%)")
            print(f"Training improvement: {train_improvement:+.2f}% ({train_perf['mean_return_pct']:.2f}% vs {baseline_train_mean:.2f}%)")
            print(f"Evaluation std change: {eval_perf['std_return_pct']:.2f}% vs {baseline_eval_std:.2f}% baseline")
        
        print(f"\n=== 10-Episode Experiment Finished Successfully ===")
        print(f"End timestamp: {datetime.now()}")
        
    except Exception as e:
        print(f"\n=== 10-Episode Experiment Failed ===")
        print(f"Error: {str(e)}")
        print(f"End timestamp: {datetime.now()}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
