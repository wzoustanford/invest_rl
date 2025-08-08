"""
Show the exact files at the indices used in the Sequential Supervised experiment
"""

# Load data files
data_dir = '/home/ubuntu/code/angle_rl/invest/data/'
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

# File indices used in the experiment
indices_used = {
    2022: [464, 507, 551, 594, 639, 682],  # Feb, Apr, Jun, Aug, Oct, Dec
    2023: [704, 747, 790, 837, 883, 929],  # Jan, Mar, May, Jul, Sep, Nov
    2024: [975, 1022]                      # Jan, Mar
}

print("="*100)
print("EXACT FILES USED IN SEQUENTIAL SUPERVISED EXPERIMENT")
print("="*100)

for year, indices in indices_used.items():
    print(f"\n{year} FILES:")
    print("-"*100)
    for idx in indices:
        if idx < len(all_files):
            filepath = all_files[idx]
            # Extract just the key parts of the filename
            import os
            filename = os.path.basename(filepath)
            # Extract dates
            import re
            pattern = r'training_data_start_date_(\d{4}_\d{2}_\d{2})_test_data_start_date_(\d{4}_\d{2}_\d{2})'
            match = re.search(pattern, filename)
            if match:
                train_date = match.group(1).replace('_', '-')
                test_date = match.group(2).replace('_', '-')
                print(f"  Index {idx:4d}: Test Date: {test_date} | Train Start: {train_date}")
                print(f"           Full path: {filepath}")
            else:
                print(f"  Index {idx:4d}: {filepath}")
        else:
            print(f"  Index {idx:4d}: OUT OF RANGE")
    
print("\n" + "="*100)
print("SUMMARY")
print("="*100)
print("""
These files were used as TEST files (where predictions were made).
For each test file, the model was trained on the 7 preceding files (indices n-6 to n).

Trading Schedule:
- 2022: Bi-monthly (Feb, Apr, Jun, Aug, Oct, Dec) - 6 trades
- 2023: Bi-monthly (Jan, Mar, May, Jul, Sep, Nov) - 6 trades  
- 2024: Bi-monthly (Jan, Mar) - 2 trades only (limited data)

Note: This was sampled bi-monthly trading (every 2 months) to speed up computation.
For true monthly (12x/year), we would use all 12 months.
""")