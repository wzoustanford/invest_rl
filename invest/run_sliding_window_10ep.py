#!/usr/bin/env python3
"""
Run 10-episode sliding window experiment for comparison with 5-episode baseline.
"""

import sys
import os
import time
import subprocess
import threading
from datetime import datetime
sys.path.append('/home/ubuntu/code/angle_rl/invest')

from sliding_window_experiment import SlidingWindowExperiment


def run_10_episode_experiment():
    """Run the 10-episode sliding window experiment with logging."""
    
    # Setup logging files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/ubuntu/code/angle_rl/invest/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    stdout_log = f"{log_dir}/sliding_window_10ep_{timestamp}_stdout.log"
    stderr_log = f"{log_dir}/sliding_window_10ep_{timestamp}_stderr.log"
    combined_log = f"{log_dir}/sliding_window_10ep_{timestamp}_combined.log"
    
    print(f"=== 10-Episode Sliding Window Experiment ===")
    print(f"Starting at: {datetime.now()}")
    print(f"Comparison with 5-episode baseline")
    print(f"Logs will be saved to:")
    print(f"  stdout: {stdout_log}")
    print(f"  stderr: {stderr_log}")
    print(f"  combined: {combined_log}")
    print(f"\nTo monitor progress, run:")
    print(f"  tail -f {combined_log}")
    
    # Create a Python script to run
    runner_script = f"""
import sys
import os
import time
from datetime import datetime

sys.path.append('/home/ubuntu/code/angle_rl/invest')
from sliding_window_experiment import SlidingWindowExperiment

def main():
    print(f"=== 10-Episode Sliding Window Experiment Started ===")
    print(f"Timestamp: {{datetime.now()}}")
    print(f"Process ID: {{os.getpid()}}")
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
    
    print(f"\\n=== Starting 10-Episode DQN Experiment ===")
    print(f"Expected improvements over 5-episode baseline:")
    print(f"  - Better model convergence")
    print(f"  - More stable evaluation performance")
    print(f"  - Reduced variance across windows")
    start_time = time.time()
    
    try:
        # Run the experiment
        results = experiment.run_full_experiment(algorithm='dqn')
        
        total_time = time.time() - start_time
        print(f"\\n=== 10-Episode Experiment Complete ===")
        print(f"Total runtime: {{total_time/3600:.2f}} hours")
        
        # Print results summary
        if 'summary' in results and 'evaluation_performance' in results['summary']:
            summary = results['summary']
            eval_perf = summary['evaluation_performance']
            train_perf = summary['training_performance']
            
            print(f"\\n=== 10-Episode Results Summary ===")
            print(f"Successful windows: {{summary['successful_windows']}}/{{summary['total_windows']}}")
            print(f"Total evaluation days: {{summary['total_eval_days']}}")
            print(f"Average evaluation return: {{eval_perf['mean_return_pct']:.2f}}% ± {{eval_perf['std_return_pct']:.2f}}%")
            print(f"Evaluation range: {{eval_perf['min_return_pct']:.2f}}% to {{eval_perf['max_return_pct']:.2f}}%")
            print(f"Average training return: {{train_perf['mean_return_pct']:.2f}}% ± {{train_perf['std_return_pct']:.2f}}%")
            print(f"Individual evaluation returns: {{[f'{{r:.1f}}%' for r in eval_perf['returns_list']]}}")
            print(f"Individual training returns: {{[f'{{r:.1f}}%' for r in train_perf['returns_list']]}}")
            
            # Comparison with 5-episode baseline
            print(f"\\n=== Comparison with 5-Episode Baseline ===")
            baseline_eval_mean = -12.56
            baseline_eval_std = 12.94
            baseline_train_mean = -7.23
            
            eval_improvement = eval_perf['mean_return_pct'] - baseline_eval_mean
            train_improvement = train_perf['mean_return_pct'] - baseline_train_mean
            
            print(f"Evaluation improvement: {{eval_improvement:+.2f}}% ({{eval_perf['mean_return_pct']:.2f}}% vs {{baseline_eval_mean:.2f}}%)")
            print(f"Training improvement: {{train_improvement:+.2f}}% ({{train_perf['mean_return_pct']:.2f}}% vs {{baseline_train_mean:.2f}}%)")
            print(f"Evaluation std change: {{eval_perf['std_return_pct']:.2f}}% vs {{baseline_eval_std:.2f}}% baseline")
        
        print(f"\\n=== 10-Episode Experiment Finished Successfully ===")
        print(f"End timestamp: {{datetime.now()}}")
        
    except Exception as e:
        print(f"\\n=== 10-Episode Experiment Failed ===")
        print(f"Error: {{str(e)}}")
        print(f"End timestamp: {{datetime.now()}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    script_path = f"{log_dir}/sliding_window_10ep_runner_{timestamp}.py"
    with open(script_path, 'w') as f:
        f.write(runner_script)
    
    print(f"\nRunner script created: {script_path}")
    
    # Run the script with logging
    cmd = [sys.executable, script_path]
    
    print(f"\nStarting 10-episode background process...")
    print(f"Command: {' '.join(cmd)}")
    
    # Start the process
    with open(stdout_log, 'w') as stdout_file, \
         open(stderr_log, 'w') as stderr_file, \
         open(combined_log, 'w') as combined_file:
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1
        )
        
        # Function to handle stdout
        def handle_stdout():
            for line in iter(process.stdout.readline, ''):
                stdout_file.write(line)
                stdout_file.flush()
                combined_file.write(f"[OUT] {line}")
                combined_file.flush()
                print(f"[OUT] {line.rstrip()}")
        
        # Function to handle stderr  
        def handle_stderr():
            for line in iter(process.stderr.readline, ''):
                stderr_file.write(line)
                stderr_file.flush()
                combined_file.write(f"[ERR] {line}")
                combined_file.flush()
                print(f"[ERR] {line.rstrip()}")
        
        # Start threads for output handling
        stdout_thread = threading.Thread(target=handle_stdout)
        stderr_thread = threading.Thread(target=handle_stderr)
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Wait for threads to finish
        stdout_thread.join()
        stderr_thread.join()
        
        process.stdout.close()
        process.stderr.close()
    
    end_time = datetime.now()
    print(f"\n=== 10-Episode Process Complete ===")
    print(f"Return code: {return_code}")
    print(f"End time: {end_time}")
    
    if return_code == 0:
        print("✅ 10-episode experiment completed successfully!")
    else:
        print("❌ 10-episode experiment failed!")
    
    print(f"\nLog files:")
    print(f"  Combined: {combined_log}")
    print(f"  Stdout: {stdout_log}")  
    print(f"  Stderr: {stderr_log}")
    
    return return_code, stdout_log, stderr_log, combined_log


if __name__ == "__main__":
    run_10_episode_experiment()