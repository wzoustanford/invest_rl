import pickle, torch 

data_list_f = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureTrue_testmodeFalse.txt', 'r')
data_list_f_new = open('/home/ubuntu/code/angle_rl/invest/data/data_list_2023-12-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureTrue_testmodeFalse_combine_trdata_exp.txt', 'w')

data_list = [] 
# dict_keys(['trainFeature', 'train_in_portfolio_series', 'trainNewsFeatures', 'all_train_tickers', 'train_margin_mask', 'testFeature', 'test_in_portfolio_series', 'testNewsFeatures', 'all_test_tickers', 'test_margin_mask'])
l = data_list_f.readline()
while l:
    data_list.append(l.strip())
    l = data_list_f.readline()
data_list_f.close()

D_last = dict() 
D_last['trainFeature'] = torch.Tensor() 
D_last['train_in_portfolio_series'] = torch.Tensor() 
news=dict(); news['embs'] = torch.Tensor() 
D_last['trainNewsFeatures'] = news 
D_last['all_train_tickers'] = [] 

for data_file in data_list: 
    D = dict()
    D_cur = pickle.load(open(data_file, 'rb'))
    D = D_cur
    processed_D_cur_train_feature = D_cur['trainFeature']
    while processed_D_cur_train_feature.shape[1] < 249:
        processed_D_cur_train_feature = torch.cat((processed_D_cur_train_feature[:, 0].unsqueeze(1), processed_D_cur_train_feature), dim = 1)
    
    D['trainFeature'] = torch.cat((D_last['trainFeature'], processed_D_cur_train_feature), dim = 0)
    processed_D_cur_train_series = D_cur['train_in_portfolio_series']
    while processed_D_cur_train_series.shape[1] < 5: 
        processed_D_cur_train_series = torch.cat((processed_D_cur_train_series, processed_D_cur_train_series[:, -1].unsqueeze(1)), dim = 1)
    D['train_in_portfolio_series'] = torch.cat((D_last['train_in_portfolio_series'], processed_D_cur_train_series), dim = 0)
    news = dict()
    news['embs'] = torch.cat((D_last['trainNewsFeatures']['embs'], D_cur['trainNewsFeatures']['embs']), dim = 0)
    D['trainNewsFeatures'] = news 
    D['all_train_tickers'] = D_last['all_train_tickers'] + D_cur['all_train_tickers'] 
    
    new_data_file = data_file.split('.pkl')[0] + '_combine_trdata_exp.pkl'
    print(f'processing and saving: {new_data_file}')
    pickle.dump(D, open(new_data_file, 'wb'))
    data_list_f_new.write(new_data_file+'\n')
    D_last = D 

data_list_f_new.close()