import torch, pdb, os
from train_sequential_step_model_consecutive import train_sequential_step_model_consecutive

def run_sequential_action_exp_train_test(exp_id, data_list_filename, model_type='iimodel', seed=1):
    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')
    print("training sequential step model: ...")
    gamma = 0.1
    train_sequential_step_model_consecutive(
        exp_id, 
        data_list_filename, 
        gamma, 
        obj_use_mean_return = False, 
        model_type = model_type, 
        steps = 750, 
        lr = 0.001, 
        device = torch.device('cuda'), 
        seed = seed, 
    ) 

if __name__=="__main__":
    dl = '/home/ubuntu/code/angle_rl/invest/data/data_list_1Y_custom_secconsec.txt' 
    d = {f"d25_k3c32h47_SeqCons_g0.1_1Y_it750_run{i}":dl for i in range(5)}
    model_type = 'iimodel'
    cnt = 9
    for k, v in d.items(): 
        run_sequential_action_exp_train_test(k, v, model_type, cnt) 
        cnt += 1
