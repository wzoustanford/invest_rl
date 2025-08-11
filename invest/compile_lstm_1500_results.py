"""Compile all LSTM 1500 iteration results"""
import json
import os

# Load all LSTM 1500 iteration results
result_files = [
    '/home/ubuntu/code/angle_rl/invest/data/lstm_1500iter_20250810_074736/lstm_1500iter_results.json',
    '/home/ubuntu/code/angle_rl/invest/data/lstm_1500iter_20250810_075135/lstm_1500iter_results.json',
    '/home/ubuntu/code/angle_rl/invest/data/lstm_1500iter_2023_2025_20250810_102208/lstm_1500iter_2023_2025_results.json'
]

all_results = {}
for file in result_files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            data = json.load(f)
            for year, results in data.items():
                if year not in all_results or 'summary' in results:
                    all_results[year] = results

# Display summary
print("LSTM 1500 Iterations - Complete Results")
print("=" * 60)
for year in sorted(all_results.keys()):
    if 'summary' in all_results[year]:
        s = all_results[year]['summary']
        print(f"\n{year}:")
        print(f"  Annual Return: {s['annual_return']:+.2f}%")
        print(f"  Win Rate: {s['win_rate']*100:.0f}%")
        print(f"  Months Traded: {s['num_months']}")
        print(f"  Best Month: {s['best_month']:+.2f}%")
        print(f"  Worst Month: {s['worst_month']:+.2f}%")
        
        # Show monthly details
        if 'trades' in all_results[year]:
            print(f"  Monthly Returns:")
            for trade in all_results[year]['trades']:
                print(f"    {trade['actual_trading_month']}: {trade['return']:+.2f}%", end="")
                if 'num_stocks' in trade:
                    print(f" ({trade['num_stocks']} stocks)", end="")
                print()

# Save compiled results
with open('/home/ubuntu/code/angle_rl/invest/data/lstm_1500iter_ALL_YEARS_compiled.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print("\nCompiled results saved to: lstm_1500iter_ALL_YEARS_compiled.json")