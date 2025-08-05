#!/usr/bin/env python3
"""
Large-scale DQN training script with logging and monitoring capabilities.
Trains on 265 files, evaluates on 60 files, with progress logging every 30 steps.
"""

import sys
import os
import subprocess
import datetime
from pathlib import Path

def setup_logging():
    """Setup logging directory and files."""
    log_dir = Path("/home/ubuntu/code/angle_rl/invest/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"dqn_large_scale_{timestamp}.log"
    
    return str(log_file)

def run_training_with_logging():
    """Run the training script with full logging."""
    
    # Setup logging
    log_file = setup_logging()
    
    print(f"=== Large Scale DQN Training ===")
    print(f"Log file: {log_file}")
    print(f"Training: 265 files")
    print(f"Evaluation: 60 files") 
    print(f"Progress logging: Every 30 episodes")
    print(f"\nTo monitor progress in real-time, run:")
    print(f"  tail -f {log_file}")
    print(f"\nStarting training...")
    
    # Change to the correct directory
    os.chdir("/home/ubuntu/code/angle_rl/invest")
    
    # Run the training script with logging
    cmd = [sys.executable, "run_dqn_real_data.py"]
    
    try:
        # Open log file for writing
        with open(log_file, 'w') as f:
            # Write header to log
            f.write(f"=== DQN Large Scale Training Log ===\n")
            f.write(f"Start time: {datetime.datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Working directory: {os.getcwd()}\n")
            f.write("=" * 50 + "\n\n")
            f.flush()
            
            # Run subprocess with output going to both console and log
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                
                # Write to console
                print(line.rstrip())
                sys.stdout.flush()
                
                # Write to log file
                f.write(line)
                f.flush()
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Write footer to log
            f.write(f"\n" + "=" * 50 + "\n")
            f.write(f"End time: {datetime.datetime.now()}\n")
            f.write(f"Return code: {return_code}\n")
        
        print(f"\nTraining completed with return code: {return_code}")
        print(f"Full log saved to: {log_file}")
        
        return return_code
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
        print(f"Partial log saved to: {log_file}")
        return 1
    except Exception as e:
        print(f"\nError during training: {e}")
        print(f"Log saved to: {log_file}")
        return 1

if __name__ == "__main__":
    exit_code = run_training_with_logging()
    sys.exit(exit_code)