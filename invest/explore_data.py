"""
Explore the structure of the financial data files.
"""

import pickle
import numpy as np
import torch

# Read first data file
data_file = "/home/ubuntu/code/angle_rl/invest/data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2020_03_25_test_data_start_date_2020_04_26_newsFeaturesFalse_alpacafracfiltered.pkl"

print(f"Loading data from: {data_file}")
with open(data_file, 'rb') as f:
    data = pickle.load(f)

print("\nData keys:")
for key in data.keys():
    print(f"  - {key}")

print("\nData shapes:")
for key, value in data.items():
    if hasattr(value, 'shape'):
        print(f"  {key}: {value.shape}")
    elif isinstance(value, list):
        print(f"  {key}: list of length {len(value)}")
    else:
        print(f"  {key}: {type(value)}")

# Look at specific fields
if 'trainFeature' in data:
    print(f"\ntrainFeature shape: {data['trainFeature'].shape}")
    print(f"trainFeature dtype: {data['trainFeature'].dtype}")

if 'train_in_portfolio_series' in data:
    print(f"\ntrain_in_portfolio_series shape: {data['train_in_portfolio_series'].shape}")
    print(f"train_in_portfolio_series dtype: {data['train_in_portfolio_series'].dtype}")
    print(f"Price range: [{data['train_in_portfolio_series'].min():.2f}, {data['train_in_portfolio_series'].max():.2f}]")

if 'all_train_tickers' in data:
    print(f"\nall_train_tickers length: {len(data['all_train_tickers'])}")
    print(f"First 10 tickers: {data['all_train_tickers'][:10]}")

# Check for test data
if 'testFeature' in data:
    print(f"\ntestFeature shape: {data['testFeature'].shape}")

if 'test_in_portfolio_series' in data:
    print(f"test_in_portfolio_series shape: {data['test_in_portfolio_series'].shape}")