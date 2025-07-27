import torch, pdb, pickle, utils, sys
from collections import deque
import numpy as np
from model.policy_model import PolicyModel
from model.value_model import ValueModel

RBSIZE = 300 

def train_RL_model(
        f_num, 
        exp_id, 
        data_list_filename, 
        start_date_idx, 
        end_date_idx_plus1, 
        eval_start_date_idx,
        eval_end_date_idx_plus1,
        gamma = 0.8, 
        obj_use_mean_return = True, 
        model_type = 'iimodel', 
        steps = 750, 
        lr = 0.001, 
        policy_training_interval = 2, 
        log_interval = 50, 
        eval_interval = 50, 
        device = torch.device('cuda'),
        seed = 1,
        num_policy_steps = 50, 
        num_epochs=2
    ): 
    """
    function with code logic to train the RL model 
    """

    torch.manual_seed(seed)
    data_id = data_list_filename.split('data_list')[1].split('.txt')[0]
    root_dir = '/home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/'
    cpu = torch.device('cpu')

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
    log_D['train_X'] = []
    log_D['eval_X'] = []

    print(f'training model fnum: {f_num}')
    model_id = 'seq_m_'+ exp_id +'_objm'+str(obj_use_mean_return)+'_steps'+str(steps)+'_lr'+str(lr)+'_mt'+model_type+'_' + f'fnum{f_num}' + '_'

    #D_list = utils.load_data_list(data_list_filename, return_count=False, load_as_pickle_dict=True, f_num=f_num)
    f_data_list = open(data_list_filename, 'r')

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

    B = 50 # trajectory batch size 
    B_p = 50

    ## [TODO] implement states and replay buffer with a limited size queue 

    #states = [] # state = [delta, w (a_t-1), l, x]  
    replay_buffer = deque([])
    #ranges = []
    #for e in range(num_epochs):
    #    ranges += list(range(start_date_idx, end_date_idx_plus1))
    #ranges += list(range(eval_start_date_idx, eval_end_date_idx_plus1))
    for i in range(start_date_idx, eval_end_date_idx_plus1): 
    #for i in ranges:
        if i % policy_training_interval == 0: 
            policy_model.train() 
            policy_optimizer.zero_grad() 

        fl = f_data_list.readline().strip()
        if not fl: 
            break
        D = pickle.load(open(fl, 'rb'))
        features = D['trainFeature']
        series = D['train_in_portfolio_series']
        tickers = D['all_train_tickers']
        del D

        ## reshuffle series tensor to the unified hash dimensions 
        indices = [] 
        mask = []
        for t in tickers: 
            if t in shuffle_dict:
                indices.append(shuffle_dict[t]) 
                mask.append(True)
            else: 
                mask.append(False)
        indices = torch.Tensor(indices).to(int) 
        
        base_frame = torch.zeros((num_tickers, series.shape[1]))
        base_frame[indices] = series[mask] 
        shuffled_series = base_frame
        
        if i == start_date_idx: 
            state = { 
                'delta': torch.ones((1, num_tickers)), 
                'action': 1.0 / num_tickers * torch.ones((1, num_tickers)), 
                'sharpe': torch.Tensor([0.0]).view(1, 1), 
                'features': None, 
                'tickers': None,
                'policy_pooled_acts': torch.zeros((1, policy_model.hidden_dim)), 
                'X': 1.0, 
                'prices': torch.zeros((num_tickers,1)), 
            }
        if i == eval_start_date_idx: 
            state['X'] = 1.0 
        
        # update features into state 
        state['features'] = features 
        state['tickers'] = tickers
        
        action, acts = policy_model(state, tickers, return_acts=True)
        action = action.to(cpu)
        policy_pooled_acts, _ = torch.max(acts, dim=0)
        policy_pooled_acts = policy_pooled_acts.view(1, -1)

        sharpe, mean_return, actual_return = compute_kday_returns(shuffled_series, action) 
        sharpe = torch.Tensor([sharpe]).view(1, 1)

        delta = torch.ones((1, num_tickers)) if i == start_date_idx else shuffled_series[:, 0]/ (state['prices'] + 1e-10)
        delta = delta.view(1, -1)
        X = torch.sum(delta * state['action'] * state['X'])
        X = X - transaction_cost(action, state['action'], X)

        if i > start_date_idx and i < end_date_idx_plus1: 
            replay = { 
                'prev_state': prev_state, 
                'sharpe': prev_state['sharpe'].to(cpu), 
                'state': state, 
            } 
            replay_buffer.append(replay) 
            del replay
            if len(replay_buffer) > RBSIZE: 
                #print('------ pop and remove left end queue element ------- ´')
                rb = replay_buffer.popleft()
                #rb['action'] = rb['action'].detach()
                #rb[''policy_pooled_acts''] = rb['policy_pooled_acts'].detatch()
                del rb
            del prev_state
        prev_state = state 
        del state
        state = {
            'delta': delta.detach().to(cpu), 
            'action': action.detach().view(1, -1).to(cpu), 
            'sharpe': sharpe.detach().to(cpu), 
            'policy_pooled_acts': policy_pooled_acts.detach().to(cpu), 
            'features': None, 
            'tickers': None,
            'X': X.detach(), 
            'prices': shuffled_series[:, 0].detach().to(cpu), 
        }
        del features, series, tickers, base_frame, shuffled_series, indices, acts, action, policy_pooled_acts
        
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

        if i > start_date_idx + B + 1 and i < end_date_idx_plus1: 

            ### train the value model ### 
            r_tensor, acts_tensor, output_tensor, acts_tensor_prev, output_tensor_prev = sample_replay_buffer_value_model(replay_buffer, B, device)

            value_model_1.train()
            value_model_2.train()

            acts_tensor = acts_tensor.detach().to(device)

            output_tensor = output_tensor.detach().to(device)

            acts_tensor_prev = acts_tensor_prev.detach().to(device)

            output_tensor_prev = output_tensor_prev.detach().to(device)

            r_tensor = r_tensor.detach().to(device)
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
                #print(f'step: {step}/{steps}, loss(vmse):{loss_val}')
                del Q1_prev, Q2_prev, Q1, Q2, y, loss
            
            del r_tensor, acts_tensor, output_tensor, acts_tensor_prev, output_tensor_prev
            ### train the policy model 
            if i % policy_training_interval == 0:
                policy_model.train()
                

                end_idx = len(replay_buffer) 
                start_idx = max(0, end_idx - 5 * B) 
                rndidx = np.random.permutation(range(start_idx, end_idx))
                #sum_Q = torch.Tensor([0.0]).to(device)

                for step in range(num_policy_steps): 
                    policy_optimizer.zero_grad()
                    for b in range(B_p): 
                        sample = replay_buffer[rndidx[b]] 
                        action, acts = policy_model(sample['state'], sample['state']['tickers'], return_acts=True) 
                        action = action.view(1, -1)
                        policy_pooled_acts, _ = torch.max(acts, dim=0) 
                        policy_pooled_acts = policy_pooled_acts.view(1, -1) 
                        Q1 = value_model_1(policy_pooled_acts, action) 
                        Q2 = value_model_2(policy_pooled_acts, action) 
                        Q = torch.minimum(Q1, Q2) 
                        loss = -1.0 * Q 
                        loss.backward()
                        loss = loss.detach()
                        #sum_Q = sum_Q + Q 
                    #sum_Q = sum_Q / B 
                    #loss = -1.0 * sum_Q 
                    #loss.backward() 
                    policy_optimizer.step() 
                    print(f"step: {step} / {num_policy_steps}, loss: {loss.item()}")
                print(f"--> updating POLICY model, sampled batch Q:{-1.0 * loss.item()}")

        """
        local_vars = locals()
        print("CPU Variable Sizes:")
        total_bytes = 0 
        for name, value in local_vars.items():
            size_bytes = sys.getsizeof(value)
            total_bytes += size_bytes
            print(f"  {name}: {size_bytes} bytes")
        print(f" total bytes: {total_bytes}")
        print(f"torch cuda mem allocated: {torch.cuda.memory_allocated()}")
        print(f"torch cuda mem cached: {torch.cuda.memory_cached()}")
        """

        #else: 
        #    Q1 = value_model_1(acts_tensor, output_tensor) 
        #    Q2 = value_model_2(acts_tensor, output_tensor) 
        #    Q = torch.minimum(Q1, Q2) 
        #    loss_val = -1 
        
        #acts_tensor_prev = acts_tensor.detach() 
        #output_tensor_prev = output_tensor.detach() 
        #prices_prev = shuffled_series[:, 0].detach()
        #action_output_prev = 
        
        if i > start_date_idx + B + 1 and i < end_date_idx_plus1: 
            log_D['loss'].append(loss_val)
            log_D['train_X'].append(X.detach())
            print('Train T-step: {} [{}/{} ({:.0f}%)]. X: {:.5f}. Loss (value mse): {:.5f}. Sharpe: {:.5f}. gamma*Q: {:.5f}. Q_prev: {:.5f}. Mean Return: {:.5f}. Actual Return: {:.5f}.'.format(
                i - start_date_idx, 
                i - start_date_idx, 
                end_date_idx_plus1 - start_date_idx,
                100. * (i - start_date_idx) / (end_date_idx_plus1 - start_date_idx), 
                X,
                loss_val, 
                sharpe.item(), 
                gamma * Q[0][0].item(),
                Q_prev[0][0].item(),
                mean_return.item(), 
                actual_return.item(), 
            ))
            del Q, Q_prev
        elif i <= start_date_idx + B + 1: 
            print(f"before B is reached, step: {i}/{start_date_idx + B + 1}")
        elif i >= end_date_idx_plus1: 
            log_D['eval_X'].append(X.detach())
            log_D['eval_mean_return'].append(mean_return.detach())
            log_D['eval_actual_return'].append(actual_return.detach())
            print('Eval T-step: {} [{}/{} ({:.0f}%)]. X: {:.5f}. Mean Return: {:.5f}. Actual Return: {:.5f}.'.format(
                i - eval_start_date_idx, 
                i - eval_start_date_idx, 
                eval_end_date_idx_plus1 - eval_start_date_idx,
                100. * (i - eval_start_date_idx) / (eval_end_date_idx_plus1 - eval_start_date_idx), 
                X,
                mean_return.item(), 
                actual_return.item(), 
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
    #action_output = action_output.to(torch.device('cuda'))
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

def sample_replay_buffer_value_model(replay_buffer, B, device):
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

        acts_tensor_prev = torch.cat((acts_tensor_prev , sample['prev_state']['policy_pooled_acts']), dim=0) 
        action_output_tensor_prev = torch.cat((action_output_tensor_prev, sample['prev_state']['action']), dim=0)
    
    return r_tensor, acts_tensor, action_output_tensor, acts_tensor_prev, action_output_tensor_prev
