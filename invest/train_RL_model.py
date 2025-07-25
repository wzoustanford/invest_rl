import torch, pdb, pickle, utils
from model.policy_model import PolicyModel
from model.value_model import ValueModel 

def train_RL_model(
        f_num,
        exp_id, 
        data_list_filename, 
        start_date_idx,
        end_date_idx_plus1,
        gamma = 0.8,
        obj_use_mean_return = True,
        model_type = 'iimodel',
        steps = 750,
        lr = 0.001, 
        policy_training_interval = 10, 
        log_interval = 50,
        eval_interval = 50 ,
        device = torch.device('cuda'),
        seed = 1,
    ): 
    """
    function with code logic to train the RL model 
    """

    torch.manual_seed(seed)
    data_id = data_list_filename.split('data_list')[1].split('.txt')[0]
    root_dir = '/home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/'

    ## -- model training log -- 
    log_D = dict()
    log_D['data_list_filename'] = data_list_filename
    log_D['steps'] = steps
    log_D['lr'] = lr
    log_D['log_interval'] = log_interval
    log_D['eval_interval'] = eval_interval
    log_D['seed'] = seed
    log_D['loss'] = []
    log_D['eval_portfolio_shares'] = []
    log_D['eval_actual_return'] = []
    log_D['eval_mean_return'] = []
    log_D['eval_stddev'] = []
    log_D['eval_sharpe'] = []
    log_D['top20_stocks'] = []
    log_D['f_num'] = f_num

    print(f'training model fnum: {f_num}')
    model_id = 'seq_m_'+ exp_id +'_objm'+str(obj_use_mean_return)+'_steps'+str(steps)+'_lr'+str(lr)+'_mt'+model_type+'_' + f'fnum{f_num}' + '_'

    D_list = utils.load_data_list(data_list_filename, return_count=False, load_as_pickle_dict=True, f_num=f_num)
    
    ## load the unified ticker hash dict from disk 
    save_pkl_name = exp_id + '_ticker_hash.pkl' 
    loadD = pickle.load(open(save_pkl_name, 'rb')) 
    shuffle_dict = loadD['hash_D'] 
    num_tickers = loadD['num_tickers'] 

    ## define the policy model 
    policy_model = PolicyModel(shuffle_dict=shuffle_dict, num_tickers=num_tickers).to(device)

    ## define the value model 
    value_model_1 = ValueModel(state_hdim = policy_model.hidden_dim, action_dim=policy_model.num_tickers).to(device)
    value_model_2 = ValueModel(state_hdim = policy_model.hidden_dim, action_dim=policy_model.num_tickers).to(device)

    # optimizers 
    policy_optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr) 
    value_optimizer_1 = torch.optim.Adam(value_model_1.parameters(), lr=lr) 
    value_optimizer_2 = torch.optim.Adam(value_model_2.parameters(), lr=lr) 

    T = len(D_list) 
    B = 10 # trajectory batch size 
        
    cpu_dev = torch.device('cpu') 

    ## [TODO] implement states and replay buffer with a limited size queue 

    states = [] # state = [delta, w (a_t-1), l, x]  
    replay_buffer = [] 

    global_index = 0
    #for epoch in range(num_epochs):
    for i in range(start_date_idx, end_date_idx_plus1): 
        if i % policy_training_interval == 0: 
            policy_model.train() 
            policy_optimizer.zero_grad() 

        features = D_list[i]['trainFeature'].to(device)
        series = D_list[i]['train_in_portfolio_series'].to(device)
        tickers = D_list[i]['all_train_tickers']
        output_tensor = torch.Tensor([]).to(device)
        acts_tensor = torch.Tensor([]).to(device)
        r_tensor = torch.Tensor([]).to(device) 

        ## reshuffle series tensor to the unified hash dimensions 
        indices = [] 
        for t in tickers: 
            indices.append(shuffle_dict[t]) 
        indices = torch.Tensor(indices).to(int) 
        
        base_frame = torch.zeros((num_tickers, series.shape[1])).to(device)
        base_frame[indices] = series
        shuffled_series = base_frame
        
        if i == start_date_idx: 
            state = {
                'delta': torch.ones((1, num_tickers)), 
                'action': 1.0 / num_tickers * torch.ones((1, num_tickers)), 
                'sharpe': 0.0,
                'features': torch.zeros(features.shape), 
                'policy_pooled_acts': torch.zeros((1, 32)), 
                'X': 1.0, 
                'prices': torch.zeros((num_tickers,1)),
            }
        #else: 
        #    state = states[global_index - 1] assum state is passed from previous step 
        
        # update features into state 
        state['features'] = features 
        
        action, acts = policy_model(state) 
        policy_pooled_acts, _ = torch.max(acts, dim=0)
        policy_pooled_acts = policy_pooled_acts.view(1, -1)

        sharpe, mean_return, actual_return = compute_kday_returns(shuffled_series, action) 
        delta = torch.ones((1, num_tickers)) if i == start_date_idx else shuffled_series[:, 0]/ state['prices'] 
        X = torch.sum(delta * state['action'] * state['X']) 
        X = X - transaction_cost(action, state['action'], X) 
        """
        next_state = { 
            'delta': delta, 
            'action': action, 
            'policy_pooled_acts': policy_pooled_acts, 
            'features': features, 
            'X': X, 
            'prices': shuffled_series[:, 0], 
        }
        """
        #states.append(next_state) 
        states.append(state) 
        global_index += 1 

        if i > start_date_idx: 
            replay = { 
                'prev_state': prev_state, 
                'action': prev_state['action'], 
                'sharpe': prev_state['sharpe'], 
                'state': state,
            } 
            replay_buffer.append(replay) 
        prev_state = state 
        state = {
            'delta': delta, 
            'action': action, 
            'sharpe': sharpe,
            'policy_pooled_acts': policy_pooled_acts, 
            'features': torch.zeros((1, 32)), 
            'X': X, 
            'prices': shuffled_series[:, 0], 
        }
                
        """
        for b in range(B): 
            output, acts = policy_model(features, tickers, delta[b], W[b], X[b], return_acts=True) 
            # assume we have a 1 dollar portfolio 
            # this is how many shares in the portfolio # gracefully, for margin trading, the rest of the objective is the same assuming b < 1
            portfolio_shares = output / torch.unsqueeze((shuffled_series[:, 0] + 1e-10), 1)
            actual_return = torch.sum(torch.unsqueeze((shuffled_series[:, -1] - shuffled_series[:, 0]), 1) * portfolio_shares)
            returns_series = torch.sum(shuffled_series[:, 1:] * portfolio_shares - torch.unsqueeze(shuffled_series[:, 0], 1) * portfolio_shares, dim=0)
            
            mean_return = torch.mean(returns_series)
            stddev = torch.std(returns_series)
            
            if obj_use_mean_return: 
                sharpe = mean_return / (stddev + 1e-10)
            else: 
                sharpe = actual_return / (stddev + 1e-10)
            
            ## concat all sharpe ratios together into trajectory batch 
            r_tensor = torch.cat((r_tensor, sharpe.view(1, -1)), dim = 0)
            
            ## concat hidden activations from the policy model into trajectory batch 
            acts, _ = torch.max(acts, dim=0)
            acts = acts.view(1, -1)
            acts_tensor = torch.cat((acts_tensor, acts), dim = 0)

            ## concat actions (policy output) into trajectory batch 
            output = output.view(1, -1)
            output_tensor = torch.cat((output_tensor, output), dim = 0)
        """

        if i > start_date_idx + B + 1: 
            r_tensor, acts_tensor, output_tensor, acts_tensor_prev, output_tensor_prev = sample_replay_buffer(replay_buffer, B)  

            value_model_1.train()
            value_model_2.train()
            acts_tensor = acts_tensor.detach()
            output_tensor = output_tensor.detach()
            acts_tensor_prev = acts_tensor_prev.detach()
            output_tensor_prev = output_tensor_prev.detach()
            r_tensor = r_tensor.detach()
            for step in range(1, steps):
                Q1_prev = value_model_1(acts_tensor_prev, output_tensor_prev) 
                Q2_prev = value_model_2(acts_tensor_prev, output_tensor_prev) 
                Q_prev = torch.minimum(Q1_prev, Q2_prev) 

                Q1 = value_model_1(acts_tensor, output_tensor) 
                Q2 = value_model_2(acts_tensor, output_tensor) 
                Q = torch.minimum(Q1, Q2) 

                y = r_tensor + gamma * Q 

                loss = torch.nn.functional.mse_loss(y, Q_prev)

                value_optimizer_1.zero_grad()
                value_optimizer_2.zero_grad()

                loss.backward()
                loss_val = loss.item()
                value_optimizer_1.step()
                value_optimizer_2.step()
                #print(f'step: {step}/{steps}, loss(vmse):{loss_val}, sharpe:{sharpe.item()},gamma*Q:{gamma*Q[0][0].item()}, Q_prev:{Q_prev[0][0].item()}')
        else: 
            Q1 = value_model_1(acts_tensor, output_tensor) 
            Q2 = value_model_2(acts_tensor, output_tensor) 
            Q = torch.minimum(Q1, Q2) 
            loss_val = -1 
        
        acts_tensor_prev = acts_tensor.detach() 
        output_tensor_prev = output_tensor.detach() 
        prices_prev = shuffled_series[:, 0].detach()
        action_output_prev = 
        
        if i > start_date_idx: 
            log_D['loss'].append(loss_val)
            print('Train T-step: {} [{}/{} ({:.0f}%)]. Loss (value mse): {:.5f}. Sharpe: {:.5f}. gamma*Q: {:.5f}. Q_prev: {:.5f}. Mean Return: {:.5f}. Actual Return: {:.5f}. Std Dev: {:.5f}'.format(
                i - start_date_idx, 
                i - start_date_idx, 
                end_date_idx_plus1 - start_date_idx,
                100. * i - start_date_idx / (end_date_idx_plus1 - start_date_idx), 
                loss_val, 
                sharpe.item(), 
                gamma * Q[0][0].item(),
                Q_prev[0][0].item(),
                mean_return.item(), 
                actual_return.item(), 
                stddev.item(), 
            ))
        """ 
        if step % eval_interval == 0:
            policy_model.eval()
            eval_features = D_list[T-1]['testFeature'].to(device)
            eval_series = D_list[T-1]['test_in_portfolio_series']
            if eval_series is not None:
                eval_series = eval_series.to(device)

            eval_output = policy_model(eval_features)

            # assume we have a 1 dollar portfolio 
            # this is how many shares in the portfolio 
            if eval_series is not None: 
                eval_portfolio_shares = eval_output / torch.unsqueeze((eval_series[:, 0] + 1e-10), 1)

                eval_actual_return = torch.sum(torch.unsqueeze((eval_series[:, -1] - eval_series[:, 0]), 1) * eval_portfolio_shares)

                eval_returns_series = torch.sum(eval_series[:, 1:] * eval_portfolio_shares - torch.unsqueeze(eval_series[:, 0], 1) * eval_portfolio_shares, dim=0)

                eval_mean_return = torch.mean(eval_returns_series)
                eval_stddev = torch.std(eval_returns_series)
                
                #eval_sharpe = eval_mean_return / eval_stddev 
                eval_sharpe = eval_actual_return / (eval_stddev + 1e-10)
                eval_loss = - eval_sharpe 
                
                _, top20_stocks_indices = torch.topk(eval_output, 20, dim=0)
                top20_stocks = []
                for i in range(len(top20_stocks_indices)):
                    top20_stocks.append(D_list[T-1]['all_test_tickers'][top20_stocks_indices[i]])
                
                print(f'--> Eval model:\tLoss (Sharpe ratio): {str(eval_loss.item())}\tMean Returns:{str(eval_mean_return.item())}\t Actual Returns: {str(eval_actual_return.item())}\tStd Dev: {str(eval_stddev.item())}') 
                print(f'--> Top 20 stocks: {str(top20_stocks)}')

            #if is_prod: 
            #    model_pt_filename = root_dir + model_id + 'step'+str(step)+'.pt' 
            #elif use_ppo:
            #    model_pt_filename = root_dir + model_id + 'ppo_step'+str(step)+'.pt' 
            #else: 
            model_pt_filename = root_dir + model_id+'_' + data_id + '_step'+str(step)+'.pt' 
            
            torch.save(policy_model.state_dict(), model_pt_filename)
            
            if eval_series is not None: 
                log_D['eval_portfolio_shares'].append(eval_portfolio_shares)
                log_D['eval_actual_return'].append(eval_actual_return.item())
                log_D['eval_mean_return'].append(eval_mean_return.item())
                log_D['eval_stddev'].append(eval_stddev.item())
                log_D['eval_sharpe'].append(eval_sharpe.item())
                log_D['top20_stocks'].append(top20_stocks)
    """
    model_log_filename = root_dir + model_id + '_'+data_id + '_log.pkl'
    pickle.dump(log_D, open(model_log_filename, 'wb'))
    
    return 

def compute_kday_returns(shuffled_series, action_output, obj_use_mean_return=True):
    # assume we have a 1 dollar portfolio 
    # this is how many shares in the portfolio # gracefully, for margin trading, the rest of the objective is the same assuming b < 1
    portfolio_shares = action_output / torch.unsqueeze((shuffled_series[:, 0] + 1e-10), 1)
    actual_return = torch.sum(torch.unsqueeze((shuffled_series[:, -1] - shuffled_series[:, 0]), 1) * portfolio_shares)
    returns_series = torch.sum(shuffled_series[:, 1:] * portfolio_shares - torch.unsqueeze(shuffled_series[:, 0], 1) * portfolio_shares, dim=0)
    
    mean_return = torch.mean(returns_series)
    stddev = torch.std(returns_series)
    
    if obj_use_mean_return: 
        sharpe = mean_return / (stddev + 1e-10)
    else: 
        sharpe = actual_return / (stddev + 1e-10)
    
    return sharpe, mean_return, actual_return 

def transaction_cost(current_ratios, previous_ratios, current_X):
    cost_ratio = 0.0015
    C = torch.sum(current_X * torch.abs(current_ratios - previous_ratios) * cost_ratio)
    return C 

def sample_replay_buffer(replay_buffer, B):
    ## random sample 
    end_idx = len(replay_buffer) 
    start_idx = max(0, end_idx - 5 * B) 
    rndidx = np.random.permutation(range(start_idx, end_idx))
    r_tensor = torch.Tensor()
    acts_tensor = torch.Tensor() 
    action_output_tensor = torch.Tensor() 
    
    acts_tensor_prev = torch.Tensor() 
    action_output_tensor_prev = torch.Tensor() 

    for b in range(B):
        sample = replay_buffer[rndidx[b]]
        r_tensor = torch.cat((r_tensor, sample['sharpe']), dim=0)

        acts_tensor = torch.cat((acts_tensor , sample['state']['policy_pooled_acts']), dim=0) 
        action_output_tensor = torch.cat((action_output_tensor, sample['state']['action']), dim=0)

        acts_tensor_prev = torch.cat((acts_tensor , sample['prev_state']['policy_pooled_acts']), dim=0) 
        action_output_tensor_prev = torch.cat((action_output_tensor, sample['prev_state']['action']), dim=0)
    
    return r_tensor, acts_tensor, action_output_tensor, acts_tensor_prev, action_output_tensor_prev 


