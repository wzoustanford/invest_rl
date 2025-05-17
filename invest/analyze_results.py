import pickle, numpy, os

#exp_id = 'apr6_reproduce_25d_strictly_after_fix_nonews_v1'
#exp_id = 'apr6_repro_25d_straft_nonews_convk32_h128_v2'
#exp_id = 'apr6_5d_8dnoint_straft_nn_v1'
#exp_id = 'apr6_repro_25d_straft_nonews_convk128_h256_v3'

exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2023_2024'
#exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2022_2023'

#exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2021_2022'
#exp_id = 'apr6_5d_8dnoint_straft_nn_v1_rerun_500it'
#exp_id = 'apr6_5d_8dnoint_straft_nn_v1_2023_2024_rr500it'
#exp_id = 'may1_5d_model_news_3months'
#exp_id = 'may1_5d_model_news_3months_no_news'
#exp_id = 'apr6_5d_8dnoint_straft_nn_v2_2425_fixshort'
#'apr6_5d_8dnoint_straft_nn_v1_21_22_rerun_c64'
#        'apr6_5d_8dnoint_straft_nn_v1_22_23_rerun_c64': '/home/ubuntu/code/angle_rl/invest/data/data_list_2021-04-04_2023-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt',
#        'apr6_5d_8dnoint_straft_nn_v1_21_22_rerun_c64': '/home/ubuntu/code/angle_rl/invest/data/data_list_2020-04-04_2022-04-04_tr360d_bs5d_8dinterval_newsFeatureFalse_testmodeFalse.txt', 

os.system(f'ls /home/ubuntu/code/angle_rl/invest/data/{exp_id}/*.pkl | sort > /home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt')
f = open(f'/home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v4/sorted_v4_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v3/sorted_v3_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v2/sorted_v2_pkls.txt', 'r')
#f = open('/home/ubuntu/code/angle_rl/invest/data/mar31_alleval_v1/sorted_v1_pkls.txt', 'r')

spread = 0.0014 * 2 # buy and sell spreads estimate from week of May 5th, 2025 on alpaca.markets 

l = f.readline()
eval_actual_returns = []

while l:
    print(l)
    D = pickle.load(open(l.strip(), 'rb'))
    eval_actual_returns.append(D['eval_actual_return'][-1])
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
