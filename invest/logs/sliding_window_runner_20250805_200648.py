
import sys
import os
import time
from datetime import datetime

sys.path.append('/home/ubuntu/code/angle_rl/invest')
from sliding_window_experiment import SlidingWindowExperiment

def main():
    print(f"=== Sliding Window Experiment Started ===")
    print(f"Timestamp: {datetime.now()}")
    print(f"Process ID: {os.getpid()}")
    print(f"Working directory: {os.getcwd()}")
    
    # Create experiment with 5 episodes per window
    experiment = SlidingWindowExperiment(
        target_eval_days=240,
        training_window_size=265,
        eval_window_size=60,
        window_shift=60,
        base_exp_id="sliding_window_bg"
    )
    
    # Print plan
    experiment.print_experiment_plan()
    
    print(f"\n=== Starting DQN Experiment ===")
    start_time = time.time()
    
    try:
        # Run the experiment
        results = experiment.run_full_experiment(algorithm='dqn')
        
        total_time = time.time() - start_time
        print(f"\n=== Experiment Complete ===")
        print(f"Total runtime: {total_time/3600:.2f} hours")
        
        # Print results summary
        if 'summary' in results and 'evaluation_performance' in results['summary']:
            summary = results['summary']
            eval_perf = summary['evaluation_performance']
            
            print(f"\n=== Results Summary ===")
            print(f"Successful windows: {summary['successful_windows']}/{summary['total_windows']}")
            print(f"Total evaluation days: {summary['total_eval_days']}")
            print(f"Average return: {eval_perf['mean_return_pct']:.2f}% ± {eval_perf['std_return_pct']:.2f}%")
            print(f"Return range: {eval_perf['min_return_pct']:.2f}% to {eval_perf['max_return_pct']:.2f}%")
            print(f"Individual returns: {[f'{r:.1f}%' for r in eval_perf['returns_list']]}")
        
        print(f"\n=== Experiment Finished Successfully ===")
        print(f"End timestamp: {datetime.now()}")
        
    except Exception as e:
        print(f"\n=== Experiment Failed ===")
        print(f"Error: {str(e)}")
        print(f"End timestamp: {datetime.now()}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
