import torch, pdb, os
from train_single_step_model import train_single_step_model

def run_single_action_exp_train_test(exp_id, data_list_filename, model_type='iimodel'):
    #exp_id = 'apr5_alleval_25d_news_nonews_v1'

    #exp_id = 'apr6_repro_25d_straft_nonews_convk128_h256_v3'
    exp_id = 'apr6_5d_8dnoint_straft_nn_v1_rerun_500it'
    #exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2023_2024'
    #exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2022_2023'
    #exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2021_2023'

    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')

    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2025_03_31_tr360d_bs25d_monthlyinterval.txt', 'r')
    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-25_2025-04-01_tr360d_bs25d_30dinterval.txt', 'r')

    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2020-03-25_2025-04-04_tr360d_bs25d_32dinterval_newsFeatureFalse_testmodeFalse.txt', 'r')
    data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-04-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 'r')
    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2022-04-04_2024-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 'r')
    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2021-04-04_2023-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 'r')
    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2020-04-04_2022-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 'r')

    data_list_f = open(data_list_filename, 'r')
    l = data_list_f.readline()
    cnt = 1
    while l:
        print('-->training model with: ' + l)
        train_single_step_model(
            exp_id,
            l.strip(),
            dropout_ratio = 0.0,
            obj_use_mean_return = True,
            steps = 500,
            lr = 0.001,
            model_type = model_type,
        )
        l = data_list_f.readline()
        cnt += 1

if __name__=="__main__":
    """
    d = {
        'apr6_5d_8dnoint_straft_nn_v2_2024_2025': '/home/ubuntu/code/angle_rl/invest/data/data_list_2023-04-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 
        'apr6_5d_8dnoint_straft_nn_v2_2023_2024': '/home/ubuntu/code/angle_rl/invest/data/data_list_2022-04-04_2024-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 
        'apr6_5d_8dnoint_straft_nn_v2_2022_2023': '/home/ubuntu/code/angle_rl/invest/data/data_list_2021-04-04_2023-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt',
        'apr6_5d_8dnoint_straft_nn_v2_2021_2022': '/home/ubuntu/code/angle_rl/invest/data/data_list_2020-04-04_2022-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 
    }
    d = {
        'apr6_5d_8dnoint_straft_nn_v1_22_23_rerun_c64': '/home/ubuntu/code/angle_rl/invest/data/data_list_2021-04-04_2023-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt',
        'apr6_5d_8dnoint_straft_nn_v1_21_22_rerun_c64': '/home/ubuntu/code/angle_rl/invest/data/data_list_2020-04-04_2022-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 
    }
    """
    d = {
        'apr6_5d_8dnoint_straft_nn_v2_2425_fixshort': '/home/ubuntu/code/angle_rl/invest/data/data_list_2023-04-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 
    }
    model_type = 'iimodelmargin'
    ## v2 model: 
    for k, v in d.items(): 
        run_single_action_exp_train_test(k, v, model_type) 