import torch, pdb, os
from train_single_step_model import train_single_step_model

data_list_file = "/home/ubuntu/code/angle_rl/invest/data/prod/data_list_tr360d_bs5d_prod.txt"

exp_id = 'prod_5d_models'
os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')

data_list_f = open(data_list_file, 'r')
l = data_list_f.readline()
print('-->training prod 5d model with: ' + l)
train_single_step_model(
    exp_id,
    l.strip(),
    dropout_ratio = 0.0,
    obj_use_mean_return = True,
    steps = 750,
    lr = 0.001,
    log_interval=250, 
    eval_interval=250,
    is_prod=True,
    model_type='iimodelwithnews',
)
