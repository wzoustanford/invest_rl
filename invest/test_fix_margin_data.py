import pickle, torch 

d = pickle.load(open('/home/ubuntu/code/angle_rl/invest/data/snp500_dict.pkl', 'rb'))
Dsnp500 = d['Dsnp500']

data_file_list = '/home/ubuntu/code/angle_rl/invest/data/data_list_2023-04-04_2025-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt'
f = open(data_file_list, 'r')
l = f.readline()
while l and len(l)>0: 
    D = pickle.load(open(l.strip(), 'rb'))
    ## get the margin mask with snp500 
    train_margin_mask = []
    for tic in D['all_train_tickers']: 
        if tic in Dsnp500:
            train_margin_mask.append(True)
        else: 
            train_margin_mask.append(False)
    D['train_margin_mask'] = torch.Tensor(train_margin_mask)
    
    test_margin_mask = []
    for tic in D['all_test_tickers']: 
        if tic in Dsnp500: 
            test_margin_mask.append(True)
        else: 
            test_margin_mask.append(False)
    D['test_margin_mask'] = torch.Tensor(test_margin_mask)
    
    pickle.dump(D, open(l.strip(), 'wb'))

    l = f.readline()
