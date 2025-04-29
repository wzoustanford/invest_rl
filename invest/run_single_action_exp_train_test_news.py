import torch, pdb, os
from train_single_step_model import train_single_step_model

exp_id = 'apr6_alleval_25d_news_v1_testing'
os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')

#data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2025_03_31_tr360d_bs25d_monthlyinterval.txt', 'r')
data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-25_2025-04-01_tr360d_bs25d_30dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')

l = data_list_f.readline()
cnt = 1
while l:
    print('-->training model with: ' + l)
    train_single_step_model(
        exp_id,
        l.strip(),
        dropout_ratio = 0.0,
        obj_use_mean_return = True,
        steps = 750,
        lr = 0.001,
        model_type= 'iimodelwithnews',
    )
    l = data_list_f.readline()
    cnt += 1

