import torch, pickle, argparse, pdb
from torch.distributions import Multinomial
from copy import deepcopy
from model.iimodel import IIMODEL, IIMODELWITHNEWS, IIMODELWITHNEWSADDTIONALNORM

def check_device_availability(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError("CUDA is not available. Please check your installation.")
        dev = torch.device("cuda")
    elif device == 'mps':
        if not torch.backends.mps.is_available():
            raise ValueError("MPS is not available. Please check your installation.")
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu") 
    return dev 

def train_single_step_model(
        exp_id,
        data_filename,
        dropout_ratio,
        obj_use_mean_return,
        steps,
        lr=0.0001,
        device='cuda',
        log_interval = 50,
        eval_interval = 50,
        is_prod = False,
        seed = 5,
        model_type = 'iimodel',
        use_ppo = False, 
    ):
    torch.manual_seed(seed)

    data_id = data_filename.split('single_step')[1].split('alpaca')[0]
    model_id = 'oneact_m_'+ exp_id +'_objm'+str(obj_use_mean_return)+'_steps'+str(steps)+'_lr'+str(lr)+'_mt'+model_type+'_'
    root_dir = '/home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/'
    
    if use_ppo: 
        K_ppo = 2000 
    
    ## -- model training log -- 
    log_D = dict()
    log_D['data_filename'] = data_filename
    log_D['dropout_ratio'] = dropout_ratio
    log_D['steps'] = steps
    log_D['lr'] = lr
    log_D['log_interval'] = log_interval
    log_D['eval_interval'] = eval_interval
    log_D['seed'] = seed
    log_D['portfolio_shares'] = []
    log_D['actual_return'] = []
    log_D['mean_return'] = []
    log_D['stddev'] = []
    log_D['sharpe'] = []
    log_D['eval_portfolio_shares'] = []
    log_D['eval_actual_return'] = []
    log_D['eval_mean_return'] = []
    log_D['eval_stddev'] = []
    log_D['eval_sharpe'] = []
    log_D['top20_stocks'] = []

    device = check_device_availability(device)

    data = pickle.load(open(data_filename, 'rb'))

    features = data['trainFeature'].to(device)
    series = data['train_in_portfolio_series'].to(device)

    eval_features = data['testFeature'].to(device)
    eval_series = data['test_in_portfolio_series']

    if eval_series is not None:
        eval_series = eval_series.to(device)

    if model_type == 'iimodelwithnews' or model_type == 'iimodelwithnewsadditionalnorm': 
        news_features = data['trainNewsFeatures']['embs'].to(device)
        eval_news_features = data['testNewsFeatures']['embs'].to(device)
        if model_type == 'iimodelwithnews':
            model = IIMODELWITHNEWS(dropout_ratio=dropout_ratio).to(device) 
        else: 
            model = IIMODELWITHNEWSADDTIONALNORM(dropout_ratio=dropout_ratio).to(device) 
    else: 
        model = IIMODEL(dropout_ratio=dropout_ratio).to(device)    
        if use_ppo:
            old_model = IIMODEL(dropout_ratio=dropout_ratio).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

    if seed != 1 and use_ppo: 
        checkpoint = torch.load(root_dir + model_id + 'ppo_step750.pt') 
        old_model.load_state_dict(checkpoint)
    
    for step in range(1, steps + 1):
        model.train()
        optimizer.zero_grad()

        if model_type == 'iimodelwithnews' or model_type == 'iimodelwithnewsadditionalnorm': 
            output, _ = model(features, news_features)
        else: 
            output = model(features)

        # assume we have a 1 dollar portfolio 
        # this is how many shares in the portfolio # gracefully, for margin trading, the rest of the objective is the same assuming b < 1
        portfolio_shares = output / torch.unsqueeze((series[:, 0] + 1e-10), 1)
        actual_return = torch.sum(torch.unsqueeze((series[:, -1] - series[:, 0]), 1) * portfolio_shares)

        returns_series = torch.sum(series[:, 1:] * portfolio_shares - torch.unsqueeze(series[:, 0], 1) * portfolio_shares, dim=0)
        
        mean_return = torch.mean(returns_series)
        max_return = torch.max(returns_series)
        stddev = torch.std(returns_series)
        
        if obj_use_mean_return: 
            #sharpe = max_return / (stddev + 1e-10)
            sharpe = mean_return / (stddev + 1e-10)
        else: 
            sharpe = actual_return / (stddev + 1e-10)

        if use_ppo:
            old_model.eval()
            old_output = old_model(features).detach()
            m = Multinomial(1, output.squeeze())
            m_old = Multinomial(1, old_output.squeeze())
            samples = torch.Tensor().to(device)
            for k in range(K_ppo): 
                sample = m_old.sample().view((len(old_output), 1))
                samples = torch.cat((samples, sample), dim = 1)
            samples_portfolio_shares = samples / (series[:, 0] + 1e-10).unsqueeze(1)
            samples_actual_return = torch.sum((series[:, -1] - series[:, 0]).unsqueeze(1) * samples_portfolio_shares, dim=0)
            
            samples_portfolio_shares = samples_portfolio_shares.unsqueeze(1)
            samples_returns_series = torch.sum((series[:, 1:].unsqueeze(2) - series[:, 0].unsqueeze(1).unsqueeze(2)) * samples_portfolio_shares, dim=0)

            samples_mean_return = torch.mean(samples_returns_series, dim = 0)
            samples_stddev = torch.std(samples_returns_series, dim = 0)

            if obj_use_mean_return: 
                samples_sharpe = samples_mean_return #/ (samples_stddev + 1e-10)
            else: 
                samples_sharpe = samples_actual_return #/ (samples_stddev + 1e-10)
            
            samples = torch.transpose(samples, 0, 1)
            p_over_q = torch.exp(m.log_prob(samples)) / torch.exp(m_old.log_prob(samples))
            eps = 0.2
            clipped_poq = torch.min(torch.Tensor([1 + eps]).to(device), p_over_q)
            clipped_poq = torch.max(torch.Tensor([1 - eps]).to(device), clipped_poq)

            obj = torch.mean(torch.min(p_over_q * samples_sharpe, clipped_poq * samples_sharpe))
            loss = - obj
        else: 
            loss = - sharpe 
        
        loss.backward()
        optimizer.step()

        if step % log_interval == 0:
            log_D['portfolio_shares'].append(portfolio_shares)
            log_D['actual_return'].append(actual_return.item())
            log_D['mean_return'].append(mean_return.item())
            log_D['stddev'].append(stddev.item()) 
            log_D['sharpe'].append(sharpe.item())
            
            print('Train step: {} [{}/{} ({:.0f}%)]\tLoss (Sharpe ratio): {:.6f}\tMean Return: {:.6f}\tActual Return: {:.6f}\tStd Dev: {:.6f}'.format(
                step, 
                step, 
                steps,
                100. * step / steps, 
                loss.item(), 
                mean_return.item(),
                actual_return.item(),
                stddev.item(),
            ))
        
        if step % eval_interval == 0:
            model.eval()
            if model_type == 'iimodelwithnews' or model_type == 'iimodelwithnewsadditionalnorm': 
                eval_output, _ = model(eval_features, eval_news_features)
            else: 
                eval_output = model(eval_features)

            # assume we have a 1 dollar portfolio 
            # this is how many shares in the portfolio 
            if eval_series is not None: 
                eval_portfolio_shares = eval_output / torch.unsqueeze((eval_series[:, 0] + 1e-10), 1)

                eval_actual_return = torch.sum(torch.unsqueeze((eval_series[:, -1] - eval_series[:, 0]), 1) * eval_portfolio_shares)

                eval_returns_series = torch.sum(eval_series[:, 1:] * eval_portfolio_shares - torch.unsqueeze(eval_series[:, 0], 1) * eval_portfolio_shares, dim=0)

                eval_mean_return = torch.mean(eval_returns_series, dim=0)
                eval_stddev = torch.std(eval_returns_series, dim=0)
                
                eval_sharpe = eval_mean_return / eval_stddev 
                #eval_sharpe = eval_actual_return / (eval_stddev + 1e-10)
                eval_loss = - eval_sharpe 
                
                _, top20_stocks_indices = torch.topk(eval_output, 20, dim=0)
                top20_stocks = []
                for i in range(len(top20_stocks_indices)):
                    top20_stocks.append(data['all_test_tickers'][top20_stocks_indices[i]])
                
                print(f'--> Eval model:\tLoss (Sharpe ratio): {str(eval_loss.item())}\tMean Returns:{str(eval_mean_return.item())}\t Actual Returns: {str(eval_actual_return.item())}\tStd Dev: {str(eval_stddev.item())}') 
                print(f'--> Top 20 stocks: {str(top20_stocks)}')
                
            if is_prod: 
                model_pt_filename = root_dir + model_id + 'step'+str(step)+'.pt' 
            elif use_ppo:
                model_pt_filename = root_dir + model_id + 'ppo_step'+str(step)+'.pt' 
            else: 
                model_pt_filename = root_dir + model_id+'_' + data_id + '_step'+str(step)+'.pt' 
            
            torch.save(model.state_dict(), model_pt_filename)
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
