import torch, pdb, os
from train_single_step_model_day_sell import train_single_step_model_day_sell

exp_id = 'jul11_25d_m_news_1Y_selldm750_zg'
os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')

#data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2025_03_31_tr360d_bs25d_monthlyinterval.txt', 'r')
#data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-25_2025-04-01_tr360d_bs25d_30dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')
data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')

l = data_list_f.readline()
cnt = 1
while l:
    print('-->training model with: ' + l)
    train_single_step_model_day_sell(
        exp_id,
        l.strip(),
        dropout_ratio = 0.0,
        obj_use_mean_return = True,
        steps = 750,
        lr = 0.001,
        log_interval = 250, 
        eval_interval=250,
        model_type= 'sell_action_model',
    )
    l = data_list_f.readline()
    cnt += 1
