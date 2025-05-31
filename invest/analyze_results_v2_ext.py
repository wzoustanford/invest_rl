import pickle, numpy, os

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

os.system(f'ls /home/ubuntu/code/angle_rl/invest/data/{exp_id}/*.pkl | sort > /home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt')
f = open(f'/home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v4/sorted_v4_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v3/sorted_v3_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v2/sorted_v2_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v1/sorted_v1_pkls.txt', 'r')

spread = 0 #0.0014 * 2 # buy and sell spreads estimate from week of May 5th, 2025 on alpaca.markets 

l = f.readline()
eval_actual_returns = []

while l:
    print(l)
    D = pickle.load(open(l.strip(), 'rb'))
    eval_actual_returns.append(D['eval_actual_return'][-5])
    l = f.readline()

eval_actual_returns = [x - spread for x in eval_actual_returns] # this line discounts for buy/sell spread

s = 0
cnt = 0
for x in eval_actual_returns:
    if not numpy.isnan(x):
        s+=x
        cnt+=1
mean_return = s / cnt
sum_return = s

print('mean_return')
print(mean_return)

print('sum_return')
print(sum_return)
base = 10000
money = base

for x in eval_actual_returns:
    if not numpy.isnan(x):
        money = money * (1 + x)
print('all years:')
print('starting money: '+str(base))
print('resulting money: '+str(money))
print(f'overall-return: {(money/base-1)*100}%')

money = base
for x in eval_actual_returns[-12:]:
    if not numpy.isnan(x): 
        money = money * (1 + x)
print('past year:')
print('starting money: '+str(base))
print('resulting money: '+str(money))
print(f'past-year return: {(money/base-1)*100}%')

print(eval_actual_returns)
