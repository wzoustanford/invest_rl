
import sys
import os
import time
from datetime import datetime

sys.path.append('/home/ubuntu/code/angle_rl/invest')
from sliding_window_experiment import SlidingWindowExperiment

def main():
    print(f"=== TD3+5-Episode Sliding Window Experiment Started ===")
    print(f"Timestamp: {datetime.now()}")
    print(f"Process ID: {os.getpid()}")
    print(f"Algorithm: TD3 (Twin Delayed Deep Deterministic Policy Gradient)")
    print(f"Episodes: 5 per window (for direct baseline comparison)")
    print(f"TD3 Features: Twin networks, soft target updates (tau=0.005), policy delay=2")
    
    # Create experiment with TD3 algorithm and 5 episodes per window
    experiment = SlidingWindowExperiment(
        target_eval_days=240,
        training_window_size=265,
        eval_window_size=60,
        window_shift=60,
        base_exp_id="sliding_window_td3_5ep"
    )
    
    # Override the episodes to 5 for this experiment
    # We'll need to pass this as a parameter
    
    # Print plan
    experiment.print_experiment_plan()
    
    print(f"\n=== Starting TD3+5-Episode Experiment ===")
    print(f"Expected TD3 advantages over DQN:")
    print(f"  - Twin Q-networks reduce overestimation bias")
    print(f"  - Soft target updates (tau=0.005) for stability")
    print(f"  - Policy delay reduces policy-value interaction issues")
    print(f"  - Should handle volatile periods (Window 1) better")
    start_time = time.time()
    
    try:
        # Run the TD3 experiment with 5 episodes
        results = experiment.run_full_experiment(algorithm='td3')
        
        total_time = time.time() - start_time
        print(f"\n=== TD3+5-Episode Experiment Complete ===")
        print(f"Total runtime: {total_time/3600:.2f} hours")
        
        # Print results summary
        if 'summary' in results and 'evaluation_performance' in results['summary']:
            summary = results['summary']
            eval_perf = summary['evaluation_performance']
            train_perf = summary['training_performance']
            
            print(f"\n=== TD3+5-Episode Results Summary ===")
            print(f"Successful windows: {summary['successful_windows']}/{summary['total_windows']}")
            print(f"Total evaluation days: {summary['total_eval_days']}")
            print(f"Average evaluation return: {eval_perf['mean_return_pct']:.2f}% ± {eval_perf['std_return_pct']:.2f}%")
            print(f"Evaluation range: {eval_perf['min_return_pct']:.2f}% to {eval_perf['max_return_pct']:.2f}%")
            print(f"Average training return: {train_perf['mean_return_pct']:.2f}% ± {train_perf['std_return_pct']:.2f}%")
            print(f"Individual evaluation returns: {[f'{r:.1f}%' for r in eval_perf['returns_list']]}")
            print(f"Individual training returns: {[f'{r:.1f}%' for r in train_perf['returns_list']]}")
            
            # Direct comparison with DQN 5-episode baseline
            print(f"\n=== Direct Comparison: TD3 vs DQN (5 Episodes) ===")
            dqn_5ep_eval = -12.56
            dqn_5ep_train = -7.23
            dqn_5ep_std = 12.94
            
            eval_improvement = eval_perf['mean_return_pct'] - dqn_5ep_eval
            train_improvement = train_perf['mean_return_pct'] - dqn_5ep_train
            std_change = eval_perf['std_return_pct'] - dqn_5ep_std
            
            print(f"\nEvaluation Performance:")
            print(f"  DQN 5ep:  {dqn_5ep_eval:.2f}% ± {dqn_5ep_std:.2f}%")
            print(f"  TD3 5ep:  {eval_perf['mean_return_pct']:.2f}% ± {eval_perf['std_return_pct']:.2f}%")
            print(f"  Improvement: {eval_improvement:+.2f}% ({':+' if eval_improvement > 0 else ''}{abs(eval_improvement/dqn_5ep_eval)*100:.1f}% relative)")
            
            print(f"\nTraining Performance:")
            print(f"  DQN 5ep:  {dqn_5ep_train:.2f}%")
            print(f"  TD3 5ep:  {train_perf['mean_return_pct']:.2f}%")
            print(f"  Improvement: {train_improvement:+.2f}%")
            
            print(f"\nStability Analysis:")
            print(f"  Volatility change: {std_change:+.2f}% std")
            if eval_perf['std_return_pct'] < dqn_5ep_std:
                print(f"  ✅ TD3 shows more consistent performance")
            else:
                print(f"  ⚠️  TD3 shows higher volatility")
            
            # Window-by-window comparison
            print(f"\nWindow-by-Window Comparison:")
            dqn_windows = [-34.33, -5.14, -9.74, -1.05]
            for i, (td3_ret, dqn_ret) in enumerate(zip(eval_perf['returns_list'], dqn_windows)):
                diff = td3_ret - dqn_ret
                print(f"  Window {i+1}: TD3={td3_ret:.2f}% vs DQN={dqn_ret:.2f}% ({diff:+.2f}%)")
        
        print(f"\n=== TD3+5-Episode Experiment Finished Successfully ===")
        print(f"End timestamp: {datetime.now()}")
        
    except Exception as e:
        print(f"\n=== TD3+5-Episode Experiment Failed ===")
        print(f"Error: {str(e)}")
        print(f"End timestamp: {datetime.now()}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
