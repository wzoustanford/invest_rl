import pickle, numpy, os, torch, pdb
import matplotlib.pyplot as plt 

overall_return_list = []
all_returns_list_list = []
num_trials = 5
for i in range(num_trials):
    exp_id = f"d25_k3c32h47_SeqCons_g0.1_1Y_it750_run{i}" #d25_k3c32h47_1Y_compSecC_run{i}" #

    os.system(f'ls /home/ubuntu/code/angle_rl/invest/data/{exp_id}/*.pkl | sort > /home/ubuntu/code/angle_rl/invest/data/{exp_id}/sorted_pkls.txt')
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

overall_return_list = torch.Tensor(overall_return_list)
all_returns_list_list = torch.Tensor(all_returns_list_list)

print('average over runs:') 
print(f'mean:{torch.mean(overall_return_list)}') 
print(f'std:{torch.std(overall_return_list)}') 

print('average over runs on each eval:') 
print(f'mean:{torch.mean(all_returns_list_list, dim=0)}') 
print(f'std:{torch.std(all_returns_list_list, dim=0)}') 

print(all_returns_list_list)
print(len(all_returns_list_list[0]))

exit()

if '4Y' in exp_id: 
    years_split_indices = [
        range(8),
        range(8, 20),
        range(20, 32),
        range(32, 44),
    ]
    all_yearly_returns_list_list = []
    for i in range(num_trials): 
        yearly_returns_list = []
        cur_returns_list = all_returns_list_list[i]
        for Y in years_split_indices:
            start_money = 100000.0 
            cur_money = start_money
            for x in Y:
                r = cur_returns_list[x]
                cur_money = cur_money * (1 + r) 
            R = cur_money / start_money - 1 
            yearly_returns_list.append(R)
        all_yearly_returns_list_list.append(yearly_returns_list)

    all_yearly_returns_list_list = torch.Tensor(all_yearly_returns_list_list)

    print('average over years:') 
    print(f'mean:{torch.mean(all_yearly_returns_list_list, dim=0)}') 
    print(f'std:{torch.std(all_yearly_returns_list_list, dim=0)}') 

    plt.figure()
    plt.errorbar(range(4), torch.mean(all_yearly_returns_list_list, dim=0), yerr=torch.std(all_yearly_returns_list_list, dim=0), fmt='-o', capsize=5, label='Returns')
    plt.title('Model Returns Over Years')
    plt.xlabel('Year')
    plt.ylabel('Returns')
    #plt.ylim(0.6, 0.9)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('4Y_each_year.png')
    plt.show()

    plt.figure()
    plt.errorbar(rangelen(all_returns_list_list[0]), torch.mean(all_returns_list_list, dim=0), yerr=torch.std(all_returns_list_list, dim=0), fmt='-o', capsize=5, label='Returns')
    plt.title('Model Returns Over Months')
    plt.xlabel('Months')
    plt.ylabel('Returns')
    #plt.ylim(0.6, 0.9)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('4Y_each_month.png')
    plt.show()
else:
    #pdb.set_trace()
    plt.figure()
    L0=60
    a= torch.mean(all_returns_list_list, dim=0)
    b=torch.std(all_returns_list_list, dim=0)
    A = a[:L0]
    B = b[:L0]
    plt.errorbar(range(L0), A, yerr=B, fmt='-o', capsize=5, label='Returns')
    #plt.errorbar(range(len(all_returns_list_list[0])), torch.mean(all_returns_list_list, dim=0), yerr=torch.std(all_returns_list_list, dim=0), fmt='-o', capsize=5, label='Returns')
    plt.title('Model Returns Over Months')
    plt.xlabel('Months')
    plt.ylabel('Returns')
    #plt.ylim(0.6, 0.9)
    plt.grid(True)
    plt.legend()
    #plt.tight_layout()
    #plt.savefig(f'1Y_each_month_single_0.png')
    #plt.show() 
        #plt.tight_layout()
        #plt.savefig(f'1Y_each_month_single_{i+1}.png')
    plt.show() 
    plt.savefig(f'1Y_each_month_{L0}points.png')
    L = 12
    N = 9
    interval_returns = []
    plt.figure()
    for i in range(L):
        #plt.figure()
        A = a[i : (L * (N-1)+i + 1) : L]
        B = b[i : (L * (N-1)+i + 1) : L]
        plt.errorbar(range(len(A)), A, yerr=B, fmt='-o', capsize=5, label='Returns')
        #plt.errorbar(range(len(all_returns_list_list[0])), torch.mean(all_returns_list_list, dim=0), yerr=torch.std(all_returns_list_list, dim=0), fmt='-o', capsize=5, label='Returns')
        plt.title('Model Returns Over Months')
        plt.xlabel('Months')
        plt.ylabel('Returns')
        #plt.ylim(0.6, 0.9)
        plt.grid(True)
        #plt.legend()
        plt.show()
        #plt.savefig(f'1Y_intervals12_each_month_{i}.png')
        money = 100000 
        for r in A: 
            money = money * (1 + r.item())
        interval_returns.append(money/100000 - 1)
    print('interval returns')
    print(interval_returns)
    print('interval ret mean')
    print(torch.mean(torch.Tensor(interval_returns)))
    print('interval ret std')
    print(torch.std(torch.Tensor(interval_returns)))
    plt.savefig(f'1Y_intervals12_combined.png')
