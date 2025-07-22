import torch, pdb, os
from train_single_step_model import train_single_step_model

def run_single_action_exp_train_test(exp_id, data_list_filename, model_type='iimodel'):
    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')
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
            steps = 750, 
            lr = 0.001,
            model_type = model_type,
            seed = cnt,
            use_ppo = False,
        )
        l = data_list_f.readline()
        cnt += 1

if __name__=="__main__":
    dl = '/home/ubuntu/code/angle_rl/invest/data/data_list_2020-03-25_2025-03-02_tr360d_bs25d_32dinterval_newsFeatureFalse_testmodeFalse.txt'
    d = {f"d25_k3c32h47_objR_4Y_run{i}":dl for i in range(5)}
    model_type = 'iimodel'
    for k, v in d.items(): 
        run_single_action_exp_train_test(k, v, model_type) 