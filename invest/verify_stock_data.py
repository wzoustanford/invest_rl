#!/usr/bin/env python3
"""
Verify stock price data against real market values by examining:
1. Ticker hash contents
2. Actual price series from data files
3. Compare with known market movements
"""

import pickle
import sys
import numpy as np
import torch

def load_ticker_hash(ticker_hash_file):
    """Load and display ticker hash contents."""
    print(f"Loading ticker hash from: {ticker_hash_file}")
    
    with open(ticker_hash_file, 'rb') as f:
        loadD = pickle.load(f)
    
    print(f"Number of tickers: {loadD['num_tickers']}")
    print(f"Keys in ticker hash: {list(loadD.keys())}")
    
    # Show first 20 tickers
    hash_D = loadD['hash_D']
    tickers = sorted(hash_D.keys(), key=lambda x: hash_D[x])[:20]
    print(f"\nFirst 20 tickers:")
    for ticker in tickers:
        print(f"  {ticker}: index {hash_D[ticker]}")
    
    return loadD

def examine_data_file(data_file_path, ticker_hash_dict, num_tickers):
    """Load and examine a specific data file."""
    print(f"\nExamining data file: {data_file_path}")
    
    with open(data_file_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Keys in data file: {list(data.keys())}")
    
    # Get series data
    if 'train_in_portfolio_series' in data:
        series = data['train_in_portfolio_series']
        print(f"Series shape: {series.shape}")
        print(f"Series type: {type(series)}")
        
        # Get tickers for this file
        if 'all_train_tickers' in data:
            tickers = data['all_train_tickers']
            print(f"Number of tickers in this file: {len(tickers)}")
            print(f"First 10 tickers: {tickers[:10]}")
            
            # Create mapping to unified hash dimensions like in the environment
            indices = []
            mask = []
            for t in tickers:
                if t in ticker_hash_dict:
                    indices.append(ticker_hash_dict[t])
                    mask.append(True)
                else:
                    mask.append(False)
            
            indices = torch.Tensor(indices).to(int)
            print(f"Valid tickers mapping to hash: {len(indices)}")
            
            # Create the reshuffled series like in environment
            base_frame = torch.zeros((num_tickers, series.shape[1]))
            base_frame[indices] = series[mask]
            shuffled_series = base_frame
            
            print(f"Shuffled series shape: {shuffled_series.shape}")
            
            # Examine some well-known tickers
            well_known_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            print(f"\nPrice data for well-known tickers:")
            
            for ticker in well_known_tickers:
                if ticker in ticker_hash_dict:
                    idx = ticker_hash_dict[ticker]
                    if idx < shuffled_series.shape[0]:
                        prices = shuffled_series[idx, :]
                        if prices.sum() > 0:  # Non-zero prices
                            print(f"  {ticker} (idx {idx}):")
                            print(f"    First price: {prices[0]:.2f}")
                            print(f"    Last price: {prices[-1]:.2f}")
                            print(f"    Price change: {((prices[-1]/prices[0] - 1)*100):+.2f}%")
                            print(f"    All prices: {prices[:5].tolist()}... (showing first 5)")
                        else:
                            print(f"  {ticker}: No price data (all zeros)")
                else:
                    print(f"  {ticker}: Not in ticker hash")
            
            return shuffled_series, tickers
    
    return None, None

def check_date_from_filename(filename):
    """Extract date from filename."""
    # Example: model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2020_03_25_test_data_start_date_2020_04_26_newsFeaturesFalse_alpacafracfiltered.pkl
    parts = filename.split('_')
    for i, part in enumerate(parts):
        if part == 'date' and i + 3 < len(parts):
            year = parts[i + 1]
            month = parts[i + 2]
            day = parts[i + 3]
            return f"{year}-{month}-{day}"
    return "Unknown"

def main():
    """Main verification function."""
    print("=== Stock Data Verification ===")
    
    # Load ticker hash
    ticker_hash_file = "/home/ubuntu/code/angle_rl/invest/rl_test_run2_ticker_hash.pkl"
    ticker_data = load_ticker_hash(ticker_hash_file)
    
    # Get data file list
    data_list_file = "/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt"
    with open(data_list_file, 'r') as f:
        data_files = [line.strip() for line in f.readlines() if line.strip()]
    
    print(f"\nFound {len(data_files)} data files")
    
    # Examine a few key files from our sliding windows
    test_indices = [0, 100, 200, 265, 325, 385]  # Key window boundaries
    
    for idx in test_indices:
        if idx < len(data_files):
            file_path = data_files[idx]
            date = check_date_from_filename(file_path)
            print(f"\n{'='*60}")
            print(f"File {idx}: Date {date}")
            
            shuffled_series, tickers = examine_data_file(
                file_path, 
                ticker_data['hash_D'], 
                ticker_data['num_tickers']
            )
            
            if shuffled_series is not None:
                # Calculate overall market movement
                non_zero_mask = shuffled_series.sum(dim=1) > 0
                valid_series = shuffled_series[non_zero_mask]
                
                if valid_series.shape[0] > 0:
                    # Calculate percentage changes for valid stocks
                    first_prices = valid_series[:, 0]
                    last_prices = valid_series[:, -1] 
                    
                    # Only calculate for stocks with positive prices
                    valid_price_mask = (first_prices > 0) & (last_prices > 0)
                    if valid_price_mask.sum() > 0:
                        valid_first = first_prices[valid_price_mask]
                        valid_last = last_prices[valid_price_mask]
                        
                        pct_changes = (valid_last / valid_first - 1) * 100
                        
                        print(f"\n  Market Analysis for {date}:")
                        print(f"    Valid stocks: {valid_price_mask.sum()}")
                        print(f"    Average return: {pct_changes.mean():.2f}%")
                        print(f"    Median return: {pct_changes.median():.2f}%")
                        print(f"    Min return: {pct_changes.min():.2f}%")
                        print(f"    Max return: {pct_changes.max():.2f}%")
                        
                        # Show distribution
                        positive_returns = (pct_changes > 0).sum().item()
                        total_returns = len(pct_changes)
                        print(f"    Positive returns: {positive_returns}/{total_returns} ({positive_returns/total_returns*100:.1f}%)")

if __name__ == "__main__":
    main()