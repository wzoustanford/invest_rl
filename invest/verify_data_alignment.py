"""
Verify data file alignment and extract exact dates for comparison
"""

import os
import pickle
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple


def extract_dates_from_filename(filename: str) -> Dict[str, str]:
    """Extract training and test dates from filename."""
    
    # Pattern: training_data_start_date_YYYY_MM_DD_test_data_start_date_YYYY_MM_DD
    pattern = r'training_data_start_date_(\d{4}_\d{2}_\d{2})_test_data_start_date_(\d{4}_\d{2}_\d{2})'
    match = re.search(pattern, filename)
    
    if match:
        train_date = match.group(1).replace('_', '-')
        test_date = match.group(2).replace('_', '-')
        return {
            'train_start': train_date,
            'test_start': test_date,
            'filename': os.path.basename(filename)
        }
    return {}


def load_and_inspect_data_file(filepath: str) -> Dict:
    """Load a data file and extract key information."""
    
    if not os.path.exists(filepath):
        return {'error': 'File not found'}
    
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        info = {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
        }
        
        # Extract dates from filename
        date_info = extract_dates_from_filename(filepath)
        info.update(date_info)
        
        # Check what's in the data
        if isinstance(data, dict):
            info['keys'] = list(data.keys())
            
            # Check for train/test data
            if 'trainFeature' in data:
                info['train_samples'] = data['trainFeature'].shape[0] if hasattr(data['trainFeature'], 'shape') else 'N/A'
            
            if 'testFeature' in data:
                info['test_samples'] = data['testFeature'].shape[0] if hasattr(data['testFeature'], 'shape') else 'N/A'
            
            if 'train_in_portfolio_series' in data:
                series = data['train_in_portfolio_series']
                if hasattr(series, 'shape'):
                    info['train_series_shape'] = series.shape
                    info['train_days'] = series.shape[1] if len(series.shape) > 1 else 'N/A'
            
            if 'test_in_portfolio_series' in data:
                series = data['test_in_portfolio_series']
                if series is not None and hasattr(series, 'shape'):
                    info['test_series_shape'] = series.shape
                    info['test_days'] = series.shape[1] if len(series.shape) > 1 else 'N/A'
            
            if 'all_train_tickers' in data:
                info['num_train_tickers'] = len(data['all_train_tickers'])
            
            if 'all_test_tickers' in data:
                info['num_test_tickers'] = len(data['all_test_tickers'])
        
        return info
        
    except Exception as e:
        return {'error': str(e), 'filepath': filepath}


def analyze_experiment_data_usage():
    """Analyze which data files were used in different experiments."""
    
    print("=" * 80)
    print("DATA FILE ALIGNMENT VERIFICATION")
    print("=" * 80)
    
    # Load data file list
    data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
    list_file = f'{data_dir}all_data_list.txt'
    
    with open(list_file, 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]
    
    print(f"\nTotal data files available: {len(all_files)}")
    
    # Analyze files for 2023 and 2024
    files_2023 = []
    files_2024 = []
    
    for filepath in all_files:
        date_info = extract_dates_from_filename(filepath)
        if date_info:
            test_year = date_info['test_start'][:4]
            if test_year == '2023':
                files_2023.append(date_info)
            elif test_year == '2024':
                files_2024.append(date_info)
    
    print(f"\nFiles with test dates in 2023: {len(files_2023)}")
    print(f"Files with test dates in 2024: {len(files_2024)}")
    
    # Check DQN experiment data usage
    print("\n" + "=" * 80)
    print("DQN/TD3 EXPERIMENT DATA USAGE")
    print("=" * 80)
    
    # The DQN experiments used specific offsets
    # 2023: offset 250, train 250-515, test 515-575
    # 2024: offset 500, train 500-765, test 765-825
    
    print("\n2023 Configuration (DQN/TD3):")
    print("  Base offset: 250 (file index)")
    print("  Training window: files 250-515 (265 files)")
    print("  Evaluation window: files 515-575 (60 files)")
    
    if len(all_files) > 575:
        # Show actual files used
        print("\n  Training data starts with:")
        for i in range(250, min(253, len(all_files))):
            info = extract_dates_from_filename(all_files[i])
            if info:
                print(f"    File {i}: test date {info['test_start']}")
        
        print("\n  Evaluation data starts with:")
        for i in range(515, min(518, len(all_files))):
            info = extract_dates_from_filename(all_files[i])
            if info:
                print(f"    File {i}: test date {info['test_start']}")
    
    print("\n2024 Configuration (DQN/TD3):")
    print("  Base offset: 500 (file index)")
    print("  Training window: files 500-765 (265 files)")
    print("  Evaluation window: files 765-825 (60 files)")
    
    if len(all_files) > 825:
        print("\n  Training data starts with:")
        for i in range(500, min(503, len(all_files))):
            info = extract_dates_from_filename(all_files[i])
            if info:
                print(f"    File {i}: test date {info['test_start']}")
        
        print("\n  Evaluation data starts with:")
        for i in range(765, min(768, len(all_files))):
            info = extract_dates_from_filename(all_files[i])
            if info:
                print(f"    File {i}: test date {info['test_start']}")
    
    # Check Sequential Supervised experiment data usage
    print("\n" + "=" * 80)
    print("SEQUENTIAL SUPERVISED EXPERIMENT DATA USAGE")
    print("=" * 80)
    
    # Check what files were actually used in the sequential experiment
    # by looking at the results
    results_file = f'{data_dir}seq_supervised_optimized/results.json'
    
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            seq_results = json.load(f)
        
        print("\nSequential Supervised Results Overview:")
        if 'annual_results' in seq_results:
            for year, data in seq_results['annual_results'].items():
                print(f"\n{year}:")
                if 'trades' in data:
                    print(f"  Number of trades: {len(data['trades'])}")
                    for trade in data['trades']:
                        print(f"    Month {trade.get('month')}: Return = {trade.get('return', 0)*100:.2f}%")
    
    # Now let's trace exactly which files were used
    print("\n" + "=" * 80)
    print("DETAILED FILE MAPPING FOR SEQUENTIAL SUPERVISED")
    print("=" * 80)
    
    # For each month we tested, find the corresponding files
    test_months_2023 = [1, 4, 7, 10]
    test_months_2024 = [1, 4]  # Only 2 months had data
    
    print("\n2023 Trades:")
    for month in test_months_2023:
        target_pattern = f"2023_{month:02d}"
        alt_pattern = f"2023-{month:02d}"
        
        matching = [f for f in all_files if target_pattern in f or alt_pattern in f]
        if matching:
            idx = all_files.index(matching[0])
            info = extract_dates_from_filename(matching[0])
            print(f"\n  Month {month}:")
            print(f"    File index: {idx}")
            print(f"    Test date: {info.get('test_start', 'N/A')}")
            print(f"    File: {os.path.basename(matching[0])[:80]}")
            
            # Show training files (7 consecutive before)
            print(f"    Training files (7 consecutive):")
            for i in range(max(0, idx-6), idx+1):
                if i < len(all_files):
                    train_info = extract_dates_from_filename(all_files[i])
                    print(f"      {i}: test date {train_info.get('test_start', 'N/A')}")
    
    print("\n2024 Trades:")
    for month in test_months_2024:
        target_pattern = f"2024_{month:02d}"
        alt_pattern = f"2024-{month:02d}"
        
        matching = [f for f in all_files if target_pattern in f or alt_pattern in f]
        if matching:
            idx = all_files.index(matching[0])
            info = extract_dates_from_filename(matching[0])
            print(f"\n  Month {month}:")
            print(f"    File index: {idx}")
            print(f"    Test date: {info.get('test_start', 'N/A')}")
            print(f"    File: {os.path.basename(matching[0])[:80]}")
            
            # Show training files
            print(f"    Training files (7 consecutive):")
            for i in range(max(0, idx-6), idx+1):
                if i < len(all_files):
                    train_info = extract_dates_from_filename(all_files[i])
                    print(f"      {i}: test date {train_info.get('test_start', 'N/A')}")
    
    # Load and inspect a sample file to understand structure
    print("\n" + "=" * 80)
    print("SAMPLE DATA FILE STRUCTURE")
    print("=" * 80)
    
    sample_files = [
        all_files[515] if len(all_files) > 515 else None,  # 2023 DQN eval start
        all_files[765] if len(all_files) > 765 else None,  # 2024 DQN eval start
    ]
    
    for filepath in sample_files:
        if filepath:
            print(f"\nInspecting: {os.path.basename(filepath)}")
            info = load_and_inspect_data_file(filepath)
            
            for key, value in info.items():
                if key not in ['filepath', 'keys']:
                    print(f"  {key}: {value}")
    
    return all_files


def create_alignment_summary():
    """Create a summary of data alignment."""
    
    all_files = analyze_experiment_data_usage()
    
    print("\n" + "=" * 80)
    print("ALIGNMENT SUMMARY")
    print("=" * 80)
    
    print("""
    KEY FINDINGS:
    
    1. DQN/TD3 Experiments:
       - Used fixed file indices (250-575 for "2023", 500-825 for "2024")
       - These indices don't necessarily align with calendar years
       - Training used 265 consecutive files, evaluation used 60 files
    
    2. Sequential Supervised Experiments:
       - Searched for files by actual date patterns in filenames
       - Used 7 consecutive files for training each model
       - Only found data for Jan/Apr/Jul/Oct 2023 and Jan/Apr 2024
    
    3. Data File Format:
       - Each file contains 360 days of training data
       - Test period is 25 days after training period
       - Files are named with both training and test start dates
    
    4. Potential Misalignment:
       - DQN "2023" might not be calendar year 2023
       - DQN "2024" might not be calendar year 2024
       - Sequential supervised used actual calendar dates
    
    RECOMMENDATION:
    Re-run experiments with explicit date-based file selection to ensure
    all methods are evaluated on exactly the same time periods.
    """)


if __name__ == "__main__":
    create_alignment_summary()