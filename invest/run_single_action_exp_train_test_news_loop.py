import torch, pdb, os
from train_single_step_model import train_single_step_model

exp_id = 'may17_cumul_5d_model_news_3m_t'
os.system('mkdir /home/ubuntu/code/angle_rl/invest/data/'+exp_id+'/')

#data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2025_03_31_tr360d_bs25d_monthlyinterval.txt', 'r')
#data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-25_2025-04-01_tr360d_bs25d_30dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')
data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')

D_last = dict() 
D_last['trainFeature'] = torch.Tensor() 
D_last['train_in_portfolio_series'] = torch.Tensor() 
news=dict(); news['embs'] = torch.Tensor() 
D_last['trainNewsFeatures'] = news 
D_last['all_train_tickers'] = [] 

l = data_list_f.readline()
cnt = 1
while l:
    print('-->training model with: ' + l)
    D_last = train_single_step_model(
        exp_id,
        l.strip(),
        dropout_ratio = 0.0,
        obj_use_mean_return = True,
        steps = 750,
        lr = 0.001,
        model_type= 'iimodelwithnews',
        D_last = D_last,
    )
    l = data_list_f.readline()
    cnt += 1

