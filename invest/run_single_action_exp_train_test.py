import torch, pdb, os
from train_single_step_model import train_single_step_model
#data_list_custom_1Y_11consecutive.txt
def run_single_action_exp_train_test(exp_id, data_list_filename, model_type='iimodel', seed=1):
    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')
    data_list_f = open(data_list_filename, 'r')
    l = data_list_f.readline()
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
            seed = seed,
            use_ppo = False,
        )
        l = data_list_f.readline()

if __name__=="__main__":
    #dl = '/home/ubuntu/code/angle_rl/invest/data/data_list_1Y_custom_secconsec.txt'
    dl = '/home/ubuntu/code/angle_rl/invest/data/data_list_2020-04-15_2025-06-03_tr360d_bs25d_32dinterval_newsFeatureFalse_testmodeFalse.txt'
    d = {f"d25_k3c32h47_4Y_4_15_run{i}":dl for i in range(8)}
    model_type = 'iimodel'
    cnt = 9
    for k, v in d.items():
        run_single_action_exp_train_test(k, v, model_type, cnt) 
        cnt += 1
    