import torch, pdb, os
from train_sequential_step_model import train_sequential_step_model

def run_sequential_action_exp_train_test(exp_id, data_list_filename, model_type='iimodel', seed=1):
    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')
    print("training sequential step model: ...")
    gamma = 0.5
    train_sequential_step_model(
        exp_id, 
        data_list_filename, 
        gamma, 
        obj_use_mean_return = True,
        model_type = model_type,
        steps = 2500,
        lr = 0.001,
        device = torch.device('cuda'),
        seed = seed,
    )

if __name__=="__main__":
    dl = '/home/ubuntu/code/angle_rl/invest/data/data_list_2023-04-05_2025-04-04_tr360d_bs25d_32dinterval_newsFeatureFalse_testmodeFalse.txt.oneless'
    d = {f"d25_k3c32h47_SeqC_1Y_g0.5_objmR_it2500_run{i}":dl for i in range(1)}
    model_type = 'iimodel'
    cnt = 0
    for k, v in d.items(): 
        run_sequential_action_exp_train_test(k, v, model_type, cnt) 
        cnt += 1
