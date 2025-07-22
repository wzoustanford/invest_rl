import torch, pdb, pickle
from model.iimodel import IIMODEL

def train_sequential_step_model_consecutive(
        exp_id, 
        data_list_filename, 
        gamma = 0.8,
        obj_use_mean_return = True,
        model_type = 'iimodel',
        steps = 750,
        lr = 0.001, 
        log_interval = 50,
        eval_interval = 50,
        device = torch.device('cuda'),
        seed = 1,
    ):
    torch.manual_seed(seed)
    data_id = data_list_filename.split('data_list')[1].split('.txt')[0]
    root_dir = '/home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/'
    
    data_list_f = open(data_list_filename, 'r')
    number_files = 0
    list_files = []
    l = data_list_f.readline().strip()
    while l: 
        list_files.append(l)
        number_files += 1
        l = data_list_f.readline().strip()

    for f_num in range(number_files): 
        train_test_data_file_name = list_files[f_num]
        train_one_sequential_step_model_consecutive(
            f_num, 
            root_dir,
            data_id,
            exp_id, 
            train_test_data_file_name,
            gamma,
            obj_use_mean_return,
            model_type,
            steps,
            lr, 
            log_interval,
            eval_interval,
            device,
            seed,
        )
    return 

def train_one_sequential_step_model_consecutive(
        f_num,
        root_dir,
        data_id,
        exp_id, 
        train_test_data_file_name, 
        gamma = 0.8,
        obj_use_mean_return = True,
        model_type = 'iimodel',
        steps = 750,
        lr = 0.001, 
        log_interval = 50,
        eval_interval = 50 ,
        device = torch.device('cuda'),
        seed = 1,
    ): 
    ## -- model training log -- 
    log_D = dict()
    log_D['train_test_data_file_name'] = train_test_data_file_name
    log_D['steps'] = steps
    log_D['lr'] = lr
    log_D['log_interval'] = log_interval
    log_D['eval_interval'] = eval_interval
    log_D['seed'] = seed
    log_D['loss_all'] = []
    log_D['eval_portfolio_shares'] = []
    log_D['eval_actual_return'] = []
    log_D['eval_mean_return'] = []
    log_D['eval_stddev'] = []
    log_D['eval_sharpe'] = []
    log_D['top20_stocks'] = []
    log_D['f_num'] = f_num

    print(f'training model fnum: {f_num}')
    model_id = 'seq_m_'+ exp_id +'_objm'+str(obj_use_mean_return)+'_steps'+str(steps)+'_lr'+str(lr)+'_mt'+model_type+'_' + f'fnum{f_num}' + '_'
    full_list_filename = '/home/ubuntu/code/angle_rl/invest/data/full_list_consecutive.txt'
    full_list_f = open(full_list_filename, 'r')
    l = full_list_f.readline().strip()
    full_list_train_test_files = []
    while l:
        full_list_train_test_files.append(l)
        l = full_list_f.readline().strip()
    ind = full_list_train_test_files.index(train_test_data_file_name)
    print(f'file found in full list: {ind}')
    D_list = [] 
    num_consecutive_steps = 6 
    for i in range(num_consecutive_steps):
        cur_train_test_file = full_list_train_test_files[ind - (num_consecutive_steps - i - 1)]
        D_list.append(pickle.load(open(cur_train_test_file, 'rb')))
    
    policy_model = IIMODEL().to(device) 
    
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr) 

    T = len(D_list)
    gamma = torch.Tensor([gamma]).to(device)
    
    for step in range(1, steps + 1):
        policy_model.train()
        optimizer.zero_grad()
        loss_all = torch.Tensor([0.0]).to(device)
        for i in range(T):
            # may need this for later 
            features = D_list[i]['trainFeature'].to(device)
            series = D_list[i]['train_in_portfolio_series'].to(device)

            output = policy_model(features)

            # assume we have a 1 dollar portfolio 
            # this is how many shares in the portfolio # gracefully, for margin trading, the rest of the objective is the same assuming b < 1
            portfolio_shares = output / torch.unsqueeze((series[:, 0] + 1e-10), 1)
            actual_return = torch.sum(torch.unsqueeze((series[:, -1] - series[:, 0]), 1) * portfolio_shares)
            returns_series = torch.sum(series[:, 1:] * portfolio_shares - torch.unsqueeze(series[:, 0], 1) * portfolio_shares, dim=0)
            
            mean_return = torch.mean(returns_series)
            stddev = torch.std(returns_series)
            
            if obj_use_mean_return: 
                sharpe = mean_return / (stddev + 1e-10)
            else: 
                sharpe = actual_return / (stddev + 1e-10)
            #denominator = stddev 
            #numerator_all = numerator_all + numerator * torch.pow(gamma, T - i - 1) 
            #denominator_all = denominator_all + denominator * torch.pow(gamma, T - i - 1) 
            loss = -1.0 * sharpe * torch.pow(gamma, T - i - 1) 
            loss_all = loss_all + loss 
        
        #loss_all = -1.0 * numerator_all / (denominator_all + 1e-10)
        loss_all.backward()
        optimizer.step()

        if step % log_interval == 0:
            log_D['loss_all'].append(loss_all.item())
            
            print('Train step: {} [{}/{} ({:.0f}%)]\tLoss (Sharpe ratio): {:.6f}\tMean Return: {:.6f}\tActual Return: {:.6f}\tStd Dev: {:.6f}'.format(
                step, 
                step, 
                steps,
                100. * step / steps, 
                loss_all.item(), 
                mean_return.item(),
                actual_return.item(),
                stddev.item(),
            ))
        
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
    
    model_log_filename = root_dir + model_id + '_'+data_id + '_log.pkl'
    pickle.dump(log_D, open(model_log_filename, 'wb'))
    
    return 
