import pickle, numpy, os, torch

#exp_id = 'may1_5d_model_news_3months_rm_ut'
#exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2023_2024'
#exp_id = 'may18_5d_model_news_3m_addnorm'
#exp_id = 'may18_5d_model_news_3m_rerun_a'
#exp_id = 'may18_5d_m_news_3m_addl1' 

#exp_id = 'may19_5d_model_news_3m_optzg'
#exp_id = 'may19_5d_m_news_3m_selldm' 
#exp_id = 'may19_5d_m_news_3m_optzg_lr01'
#exp_id = 'may19_5d_m_news_3m_optzg_lr0001'
#exp_id = 'may19_5d_m_news_3m_nozg_lr001'
#exp_id = 'may19_5d_m_news_3m_zg_lr001'
exp_id = 'may19_5d_m_news_3m_zg_lrtune'

exp_id = 'may1_5d_model_3months_nzg'
exp_id = 'may1_5d_model_3months_zg'
exp_id = 'may1_5d_model_3months_zg_0.5m'
exp_id = 'may1_5d_model_3months_zg_0.5m_0.999m2'
exp_id = 'may19_5d_m_news_3m_zg_0.5m_0.999m2'
exp_id = 'may19_5d_m_news_3m_nzg_again'
exp_id = 'may19_5d_m_news_3m_nzg_aga_lr0.001'
exp_id = 'may1_5d_model_3months_nnb_nzg_again'
exp_id = 'may1_5d_model_3months_nnb_nzg_again_ag'
exp_id = 'may1_5d_model_3months_nnb_nzg_again_ag_ag'

exp_id = 'may1_5d_model_3months_short_nn_nzg'
exp_id = 'may1_5d_model_3months_short_nn_ZG_tanh'
exp_id = 'may1_5d_model_3months_short_nn_ZG_softplus'
exp_id = 'may1_5d_model_3m_short_nn_ZG_softplus_maxnorm'
exp_id = 'may1_5d_model_3m_short_nn_ZG_softplus_l1norm'
exp_id = 'may1_5dm_3m_short_nn_ZG_softp_maxnorm_c32h48'
exp_id = 'may1_5dm_3m_short_nn_nzg_prod'
exp_id = 'may1_5dm_3m_short_nn_nzg_prod_softp'
exp_id = 'may1_5dm_3m_short_nn_zg_curbest_c32h48'
exp_id = 'may1_5dm_3m_short_nn_zg_curbest_c32h48_dr0.1'
exp_id = 'may1_5dm_3m_short_nn_zg_curbest_2convL_dr0.1'
exp_id = 'may1_5dm_3m_short_nn_zg_curbest_2convL_dr0.1_1250it'
exp_id = 'may1_5dm_3m_short_nn_zg_curbest_2convL_dr0.1_1750it'
exp_id = 'may1_5dm_3m_short_nn_zg_curbest_2convL_dr0.3_1250it'
exp_id = 'may1_5dm_3m_short_nn_zg_curbest_2convL_dr0.5_1250it'
exp_id = 'may1_5dm_3m_short_nn_zg_curb_2convL_lg_dr0.5_2000it'

#    '5dm_3mS_news_zg_curb_2co_dr0.1_ad05m09m2_r8',

exp_id_list = [
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r1',
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r2',
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r3',
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r4',
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r5',
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r6',
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r7',
    '5dm_3mA_news_nzg_prd_dr0_mp20_3.5kit_r8',
]
end_iter = 2000 
iter_list = [
    3500,
    3250,
    3000,
    2750,
    2500,
    2250,
    2000,
    1500,
    1250,
    1000,
    750,
    500,
]

spread = 0 #0.0014 * 2 # buy and sell spreads estimate from week of May 5th, 2025 on alpaca.markets 
all_means = []
all_stds = []
for iter in iter_list:
    idx = int(9-iter/250)
    iter_returns_list = [] 
    for exp_id in exp_id_list: 
        eval_actual_returns = []

        os.system(f'ls /home/ubuntu/code/angle_rl/invest/data/{exp_id}/*.pkl | sort > /home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt')
        f = open(f'/home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt', 'r')
        l = f.readline()
        while l:
            print(l)
            D = pickle.load(open(l.strip(), 'rb'))
            eval_actual_returns.append(D['eval_actual_return'][-idx])
            l = f.readline()

        eval_actual_returns = [x - spread for x in eval_actual_returns] # this line discounts for buy/sell spread

        base = 10000
        money = base

        for x in eval_actual_returns:
            if not numpy.isnan(x):
                money = money * (1 + x)
        exp_id_return = (money/base-1)
        print('all years:')
        print('starting money: '+str(base))
        print('resulting money: '+str(money))
        print(f'overall-return: {exp_id_return*100}%')
        iter_returns_list.append(exp_id_return)
    iter_mean = torch.mean(torch.Tensor(iter_returns_list)).item()
    iter_std = torch.std(torch.Tensor(iter_returns_list)).item()
    all_means.append(iter_mean)
    all_stds.append(iter_std)

for i in range(len(iter_list)):
    print(f"iterations: {iter_list[i]}, return_mean: {all_means[i]}, return_std: {all_stds[i]}")
