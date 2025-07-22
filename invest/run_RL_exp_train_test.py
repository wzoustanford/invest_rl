import torch, pdb, os
from train_RL_model import train_RL_model
from aggregate_tickers_RL import aggregate_tickers_RL

# use 267 days to train, and the next 100 days for validation 

def run_RL_exp_train_test(exp_id, data_list_filename, model_type='iimodel', seed=1): 
    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/') 
    print("training sequential step model: ...") 
    train_start_idx = 0 
    train_end_idx_plus1 = 267 
    test_start_idx = train_end_idx_plus1 
    test_end_idx_plus1 = test_start_idx + 100 
    data_list = [] 
    f= open(data_list_filename, 'r') 
    l = f.readline() 
    while l: 
        data_list.append(l.strip()) 
        l = f.readline() 
    f.close() 
    aggregate_tickers_RL(data_list, train_start_idx, train_end_idx_plus1, exp_id) 
    gamma = 0.5
    train_RL_model(
        exp_id, 
        data_list_filename, 
        train_start_idx, 
        train_end_idx_plus1, 
        gamma, 
        obj_use_mean_return = True, 
        model_type = model_type, 
        steps = 100, 
        lr = 0.001, 
        device = torch.device('cuda'), 
        seed = seed, 
    ) 

if __name__=="__main__":
    dl = '/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt' 
    d = {f"rl_test_run{i}":dl for i in range(5)}
    model_type = 'iimodel'
    cnt = 9
    for k, v in d.items(): 
        run_RL_exp_train_test(k, v, model_type, cnt) 
        cnt += 1
