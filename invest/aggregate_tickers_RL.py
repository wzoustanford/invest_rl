import os, sys, torch, pickle, pdb, bidict, utils

if __name__=="__main__": 
    utils.aggregate_tickers_RL(
        [
            '/home/ubuntu/code/angle_rl/invest/data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2021_03_13_test_data_start_date_2021_04_14_newsFeaturesFalse_alpacafracfiltered.pkl',
            '/home/ubuntu/code/angle_rl/invest/data/model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_2021_03_14_test_data_start_date_2021_04_15_newsFeaturesFalse_alpacafracfiltered.pkl',
        ],
        0,
        2,
        'testing_'
    )
