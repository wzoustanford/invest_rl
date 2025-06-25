import pickle, numpy, os, torch, pdb

overall_return_list = []
all_returns_list_list = []
for i in range(4):
    exp_id = f"jun24_25d_ppo_run{i}"

    os.system(f'ls /home/ubuntu/code/angle_rl/invest/data/{exp_id}/*1250*.pkl | sort > /home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt')
    f = open(f'/home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt', 'r')
    spread = 0

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
    ret = (money/base-1) 
    print(eval_actual_returns)
    overall_return_list.append(ret)
    all_returns_list_list.append(eval_actual_returns)

pdb.set_trace()
overall_return_list = torch.Tensor(overall_return_list)
all_returns_list_list = torch.Tensor(all_returns_list_list)

print('average over runs:') 
print(f'mean:{torch.mean(overall_return_list)}')
print(f'std:{torch.std(overall_return_list)}')

print('average over runs on each eval:') 
print(f'mean:{torch.mean(all_returns_list_list, dim=0)}')
print(f'std:{torch.std(all_returns_list_list, dim=0)}')

print(all_returns_list_list)