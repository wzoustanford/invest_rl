#!/usr/bin/env python3
"""
Background runner for sliding window experiment with logging.
This runs the experiment in background with stdout/stderr capture.
"""

import sys
import os
import time
import subprocess
import threading
from datetime import datetime
sys.path.append('/home/ubuntu/code/angle_rl/invest')

from sliding_window_experiment import SlidingWindowExperiment


def run_with_logging():
    """Run the sliding window experiment with proper logging."""
    
    # Setup logging files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"/home/ubuntu/code/angle_rl/invest/logs"
    os.makedirs(log_dir, exist_ok=True)
    
    stdout_log = f"{log_dir}/sliding_window_{timestamp}_stdout.log"
    stderr_log = f"{log_dir}/sliding_window_{timestamp}_stderr.log"
    combined_log = f"{log_dir}/sliding_window_{timestamp}_combined.log"
    
    print(f"=== Background Sliding Window Experiment ===")
    print(f"Starting at: {datetime.now()}")
    print(f"Logs will be saved to:")
    print(f"  stdout: {stdout_log}")
    print(f"  stderr: {stderr_log}")
    print(f"  combined: {combined_log}")
    print(f"\nTo monitor progress, run:")
    print(f"  tail -f {combined_log}")
    print(f"  # or")
    print(f"  tail -f {stdout_log}")
    
    # Create a Python script to run
    runner_script = f"""
import sys
import os
import time
from datetime import datetime

sys.path.append('/home/ubuntu/code/angle_rl/invest')
from sliding_window_experiment import SlidingWindowExperiment

def main():
    print(f"=== Sliding Window Experiment Started ===")
    print(f"Timestamp: {{datetime.now()}}")
    print(f"Process ID: {{os.getpid()}}")
    print(f"Working directory: {{os.getcwd()}}")
    
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
    
    print(f"\\n=== Starting DQN Experiment ===")
    start_time = time.time()
    
    try:
        # Run the experiment
        results = experiment.run_full_experiment(algorithm='dqn')
        
        total_time = time.time() - start_time
        print(f"\\n=== Experiment Complete ===")
        print(f"Total runtime: {{total_time/3600:.2f}} hours")
        
        # Print results summary
        if 'summary' in results and 'evaluation_performance' in results['summary']:
            summary = results['summary']
            eval_perf = summary['evaluation_performance']
            
            print(f"\\n=== Results Summary ===")
            print(f"Successful windows: {{summary['successful_windows']}}/{{summary['total_windows']}}")
            print(f"Total evaluation days: {{summary['total_eval_days']}}")
            print(f"Average return: {{eval_perf['mean_return_pct']:.2f}}% ± {{eval_perf['std_return_pct']:.2f}}%")
            print(f"Return range: {{eval_perf['min_return_pct']:.2f}}% to {{eval_perf['max_return_pct']:.2f}}%")
            print(f"Individual returns: {{[f'{{r:.1f}}%' for r in eval_perf['returns_list']]}}")
        
        print(f"\\n=== Experiment Finished Successfully ===")
        print(f"End timestamp: {{datetime.now()}}")
        
    except Exception as e:
        print(f"\\n=== Experiment Failed ===")
        print(f"Error: {{str(e)}}")
        print(f"End timestamp: {{datetime.now()}}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
"""
    
    script_path = f"{log_dir}/sliding_window_runner_{timestamp}.py"
    with open(script_path, 'w') as f:
        f.write(runner_script)
    
    print(f"\nRunner script created: {script_path}")
    
    # Run the script with logging
    cmd = [sys.executable, script_path]
    
    print(f"\nStarting background process...")
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
    print(f"\n=== Process Complete ===")
    print(f"Return code: {return_code}")
    print(f"End time: {end_time}")
    
    if return_code == 0:
        print("✅ Experiment completed successfully!")
    else:
        print("❌ Experiment failed!")
    
    print(f"\nLog files:")
    print(f"  Combined: {combined_log}")
    print(f"  Stdout: {stdout_log}")  
    print(f"  Stderr: {stderr_log}")
    
    return return_code, stdout_log, stderr_log, combined_log


def quick_runtime_test():
    """Run a quick test to estimate runtime with 5 episodes."""
    print("=== Quick Runtime Test (5 episodes) ===")
    
    experiment = SlidingWindowExperiment(
        training_window_size=5,  # Mini window
        eval_window_size=3,
        window_shift=2,
        target_eval_days=3,  # Just 1 window
        base_exp_id="runtime_test"
    )
    
    start_time = time.time()
    result = experiment.run_window_experiment(experiment.windows[0], algorithm='dqn')
    test_time = time.time() - start_time
    
    if 'error' not in result:
        # Scale up for full experiment (5 episodes)
        full_experiment = SlidingWindowExperiment()
        num_windows = len(full_experiment.windows)
        
        # Scaling factors
        data_scale = (265 + 60) / (5 + 3)  # ~40x more data
        episode_scale = 1.0  # Same 5 episodes
        
        scaling_factor = data_scale * episode_scale
        estimated_time_per_window = test_time * scaling_factor
        estimated_total_time = estimated_time_per_window * num_windows
        
        print(f"Test time: {test_time:.1f}s")
        print(f"Estimated time per full window: {estimated_time_per_window/60:.1f} minutes")
        print(f"Estimated total time: {estimated_total_time/3600:.1f} hours")
        print(f"Number of windows: {num_windows}")
        
        return estimated_total_time
    else:
        print(f"Test failed: {result['error']}")
        return None


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_runtime_test()
    else:
        run_with_logging()