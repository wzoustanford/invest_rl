"""
Display exactly which files were used for 2024 trading in Sequential Supervised experiment
"""

import os

def show_2024_sequential_files():
    """Show the exact files used for 2024 trading."""
    
    # Load the data file list
    data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
    list_file = f'{data_dir}all_data_list.txt'
    
    with open(list_file, 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]
    
    print("=" * 100)
    print("EXACT FILES USED FOR 2024 TRADING - SEQUENTIAL SUPERVISED ALGORITHM")
    print("=" * 100)
    
    # Based on the run_optimized_sequential_experiment.py output:
    # - Month 1 (January 2024): File index 975
    # - Month 4 (April 2024): File index 1044
    
    print("\n2024 MONTH 1 (JANUARY) TRADE:")
    print("-" * 100)
    
    # January 2024 trade
    jan_test_idx = 975
    print(f"\nTEST FILE (where predictions are made):")
    print(f"  Index {jan_test_idx}: {all_files[jan_test_idx]}")
    
    print(f"\nTRAINING FILES (7 consecutive days used to train the model):")
    for i in range(jan_test_idx - 6, jan_test_idx + 1):
        print(f"  Index {i}: {all_files[i]}")
    
    print("\n" + "=" * 100)
    print("\n2024 MONTH 4 (APRIL) TRADE:")
    print("-" * 100)
    
    # April 2024 trade
    apr_test_idx = 1044
    print(f"\nTEST FILE (where predictions are made):")
    print(f"  Index {apr_test_idx}: {all_files[apr_test_idx]}")
    
    print(f"\nTRAINING FILES (7 consecutive days used to train the model):")
    for i in range(apr_test_idx - 6, apr_test_idx + 1):
        print(f"  Index {i}: {all_files[i]}")
    
    print("\n" + "=" * 100)
    print("SUMMARY:")
    print("-" * 100)
    print("""
    For 2024 trading, the Sequential Supervised algorithm:
    
    1. JANUARY 2024 TRADE:
       - Trained on 7 files from Dec 25, 2023 to Jan 1, 2024
       - Made predictions for Jan 1, 2024 test period (25-day holding)
       - Result: +0.95% return
    
    2. APRIL 2024 TRADE:  
       - Trained on 7 files from Mar 24, 2024 to Apr 1, 2024
       - Made predictions for Apr 1, 2024 test period (25-day holding)
       - Result: -9.22% return
    
    3. MONTHS 7 and 10 (July and October 2024):
       - NO DATA AVAILABLE (files don't exist yet)
       - This is why only 2 trades were executed for 2024
    
    Note: Each file contains:
    - 360 days of training data (ending at train_data_start_date)
    - 25 days of test data (starting at test_data_start_date)
    - The model is trained on the training portion and evaluated on the test portion
    """)

if __name__ == "__main__":
    show_2024_sequential_files()