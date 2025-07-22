import os, sys, torch, pickle, pdb, bidict

def aggregate_tickers_RL(data_file_list, start_idx, end_idx_plus1, exp_id): 
    save_pkl_name = exp_id + '_ticker_hash.pkl' 
    cnt = 0
    hash_D = dict()
    for idx in range(start_idx, end_idx_plus1): 
        filename = data_file_list[idx] 
        f = open(filename, 'rb') 
        D = pickle.load(f) 
        test_features = D['trainFeature'] 
        all_train_tickers = D['all_train_tickers'] 
        for ticker in all_train_tickers:
            if ticker not in hash_D:
                hash_D[ticker] = cnt 
                cnt += 1
    print('total number of tickers: ')
    print(cnt)
    saveD = dict() 
    saveD['hash_D'] = hash_D 
    saveD['num_tickers'] = cnt 
    pickle.dump(saveD, open(save_pkl_name, 'wb'))

if __name__=="__main__": 
    aggregate_tickers_RL(
        [
            '/home/ubuntu/code/angle_rl/invest/data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2021_03_13_test_data_start_date_2021_04_14_newsFeaturesFalse_alpacafracfiltered.pkl',
            '/home/ubuntu/code/angle_rl/invest/data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2021_03_14_test_data_start_date_2021_04_15_newsFeaturesFalse_alpacafracfiltered.pkl',
        ],
        0,
        2,
        'testing_'
    )
