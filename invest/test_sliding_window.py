#!/usr/bin/env python3
"""
Test script for sliding window experiment.
This will run a minimal version to test the implementation and estimate runtime.
"""

import sys
import time
sys.path.append('/home/ubuntu/code/angle_rl/invest')

from sliding_window_experiment import SlidingWindowExperiment


def test_window_calculation():
    """Test window calculation logic."""
    print("=== Testing Window Calculation ===")
    
    # Create a small test experiment
    experiment = SlidingWindowExperiment(
        training_window_size=10,
        eval_window_size=5,
        window_shift=3,
        target_eval_days=15  # Should create 3 windows
    )
    
    print(f"Total files available: {experiment.total_files}")
    print(f"Calculated windows: {len(experiment.windows)}")
    
    for window in experiment.windows:
        print(f"  Window {window['window_id']}: "
              f"Train[{window['train_start']}:{window['train_end']}] → "
              f"Eval[{window['eval_start']}:{window['eval_end']}]")
    
    return len(experiment.windows) == 3


def test_single_window():
    """Test running a single window with minimal parameters."""
    print("\n=== Testing Single Window (Mini) ===")
    
    # Create experiment with very small parameters for testing
    experiment = SlidingWindowExperiment(
        training_window_size=5,  # Just 5 training days
        eval_window_size=3,      # Just 3 evaluation days 
        window_shift=2,
        target_eval_days=3,      # Just 1 window
        base_exp_id="test_mini"
    )
    
    if not experiment.windows:
        print("No windows calculated!")
        return False
    
    print(f"Testing window: {experiment.windows[0]}")
    
    start_time = time.time()
    
    try:
        # Run just the first window with very reduced parameters
        result = experiment.run_window_experiment(
            experiment.windows[0], 
            algorithm='dqn'
        )
        
        test_time = time.time() - start_time
        
        if 'error' in result:
            print(f"Test failed with error: {result['error']}")
            return False
        
        print(f"Test successful!")
        print(f"Training return: {result['training']['return_pct']:.2f}%")
        print(f"Evaluation return: {result['evaluation']['return_pct']:.2f}%")
        print(f"Test runtime: {test_time:.1f} seconds")
        
        # Estimate full runtime
        full_experiment = SlidingWindowExperiment()
        num_full_windows = len(full_experiment.windows)
        
        # Scale up the time estimate (full windows are much larger)
        # Mini window: 5 train + 3 eval = 8 days
        # Full window: 265 train + 60 eval = 325 days
        # So roughly 40x more data per window
        # Plus full training uses 30 episodes vs our reduced parameters
        
        scaling_factor = (325 / 8) * (30 / 5)  # data scaling * episode scaling
        estimated_time_per_full_window = test_time * scaling_factor
        estimated_total_time = estimated_time_per_full_window * num_full_windows
        
        print(f"\n=== Runtime Estimates ===")
        print(f"Mini window time: {test_time:.1f}s")
        print(f"Estimated scaling factor: {scaling_factor:.1f}x")
        print(f"Estimated time per full window: {estimated_time_per_full_window/60:.1f} minutes")
        print(f"Full windows needed: {num_full_windows}")
        print(f"Estimated total experiment time: {estimated_total_time/3600:.1f} hours")
        
        return True
        
    except Exception as e:
        test_time = time.time() - start_time
        print(f"Test failed after {test_time:.1f}s with error: {str(e)}")
        return False


def test_experiment_plan():
    """Test the full experiment plan without running."""
    print("\n=== Testing Full Experiment Plan ===")
    
    experiment = SlidingWindowExperiment(
        target_eval_days=240,
        training_window_size=265,
        eval_window_size=60,
        window_shift=60
    )
    
    experiment.print_experiment_plan()
    
    # Check for any obvious issues
    issues = []
    
    if len(experiment.windows) == 0:
        issues.append("No windows calculated")
    
    for window in experiment.windows:
        if window['eval_end'] > experiment.total_files:
            issues.append(f"Window {window['window_id']} needs {window['eval_end']} files but only {experiment.total_files} available")
    
    if issues:
        print(f"\nIssues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print(f"\nPlan validation: OK")
        return True


def main():
    """Run all tests."""
    print("=== Sliding Window Experiment Tests ===\n")
    
    tests = [
        ("Window Calculation", test_window_calculation),
        ("Experiment Plan", test_experiment_plan),
        ("Single Window Runtime", test_single_window)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        print("-" * 50)
        
        try:
            start = time.time()
            success = test_func()
            duration = time.time() - start
            
            results[test_name] = {
                'success': success,
                'duration': duration
            }
            
            status = "PASS" if success else "FAIL"
            print(f"{test_name}: {status} ({duration:.1f}s)")
            
        except Exception as e:
            results[test_name] = {
                'success': False,
                'error': str(e),
                'duration': time.time() - start
            }
            print(f"{test_name}: ERROR - {str(e)}")
    
    # Summary
    print(f"\n=== Test Summary ===")
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        print(f"  {test_name}: {status}")
        if 'error' in result:
            print(f"    Error: {result['error']}")
    
    return all(r['success'] for r in results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)