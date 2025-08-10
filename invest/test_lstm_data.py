"""
Test LSTM data preparation
"""

import torch
import pickle
import numpy as np
import os

data_dir = '/home/ubuntu/code/angle_rl/invest/data/'

# Load file list
with open(f'{data_dir}all_data_list.txt', 'r') as f:
    all_files = [line.strip() for line in f if line.strip()]

print(f"Total files: {len(all_files)}")

# Test loading first 20 files
valid_files = []
for i in range(min(20, len(all_files))):
    fpath = all_files[i]
    if os.path.exists(fpath):
        try:
            with open(fpath, 'rb') as f:
                data = pickle.load(f)
            
            if 'trainFeature' in data and data['trainFeature'] is not None:
                valid_files.append(i)
                if len(valid_files) <= 3:
                    print(f"\nFile {i}: {os.path.basename(fpath)}")
                    print(f"  trainFeature shape: {data['trainFeature'].shape}")
                    print(f"  train_in_portfolio_series shape: {data.get('train_in_portfolio_series').shape if 'train_in_portfolio_series' in data else 'None'}")
        except Exception as e:
            print(f"Error loading file {i}: {e}")

print(f"\nValid files found: {len(valid_files)}")

# Test sequence preparation
if len(valid_files) >= 8:
    print("\nTesting sequence preparation with first 8 valid files...")
    
    sequences = []
    for i in range(len(valid_files) - 7):
        seq_features = []
        
        # Load 7 consecutive files
        for j in range(7):
            file_idx = valid_files[i + j]
            with open(all_files[file_idx], 'rb') as f:
                data = pickle.load(f)
            
            features = data['trainFeature'].cpu().numpy()
            # Take mean across stocks for simplicity
            avg_features = np.mean(features, axis=0)
            seq_features.append(avg_features)
        
        sequences.append(np.array(seq_features))
    
    if sequences:
        sequences = np.array(sequences)
        print(f"Created sequences shape: {sequences.shape}")
        print(f"Expected: (num_sequences, sequence_length=7, feature_dim)")