import torch, pdb, os
from train_single_step_model import train_single_step_model

#exp_id = 'may1_5dm_3m_short_nn_ZG_softp_maxnorm_c32h48'
for i in range(8):
    exp_id = f'5dm_3mA_news_nzg_prd_dr0_h254_3.5kit_r{i+1}'#f'5dm_3mS_news_zg_curb_2co_dr0.1_ad05m09m2_r{i+1}'
    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')
    
    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2025_03_31_tr360d_bs25d_monthlyinterval.txt', 'r')
    #data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-25_2025-04-01_tr360d_bs25d_30dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')
    data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')
    
    l = data_list_f.readline()
    cnt = 1
    while l:
        print('-->training model with: ' + l)
        train_single_step_model(
            exp_id,
            l.strip(),
            dropout_ratio = 0.0,
            obj_use_mean_return = True,
            steps = 3500,
            lr = 0.001,
            model_type= 'iimodelwithnews',
            log_interval = 250,
            eval_interval = 250,
        )
        l = data_list_f.readline()
        cnt += 1

