import torch, pdb, os, utils 
from train_RL_model import train_RL_model

def run_RL_exp_train_test(exp_id, data_list_filename, model_type='iimodel', seed=1): 
    os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/') 
    print("training RL model: ...") 

    # use 267 days to train, and the next 100 days for validation 
    train_start_idx = 0 
    train_end_idx_plus1 = 267 
    test_start_idx = train_end_idx_plus1 
    test_end_idx_plus1 = test_start_idx + 100 

    data_list = utils.load_data_list(data_list_filename)

    ## a custom step to aggregate the tickers into a uniform hash, so that action space is unified across the training data 
    utils.aggregate_tickers_RL(data_list, train_start_idx, train_end_idx_plus1, exp_id) 

    gamma = 0.5
    f_num = 370
    train_RL_model(
        f_num,
        exp_id, 
        data_list_filename, 
        train_start_idx, 
        train_end_idx_plus1, 
        gamma, 
        obj_use_mean_return = True, 
        model_type = model_type, 
        steps = 30, 
        lr = 0.001, 
        device = torch.device('cuda'), 
        seed = seed, 
    ) 

if __name__=="__main__":
    ### execution of script. There is an option to have multiple runs with different random seeds. 

    data_list = '/home/ubuntu/code/angle_rl/invest/data/all_data_list.txt' 
    D = {f"rl_test_run{i}":data_list for i in range(5)}
    model_type = 'iimodel'

    seed = 9 
    for k, v in D.items(): 
        run_RL_exp_train_test(k, v, model_type, seed) 
        seed += 1
