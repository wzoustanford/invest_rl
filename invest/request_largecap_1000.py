import pickle, pdb
from utils import get_finance_api_data 

url = 'https://financialmodelingprep.com/stable/company-screener?marketCapMoreThan=35000000000&limit=2000'
large_cap_data = get_finance_api_data(url=url, key_only=False)
print(len(large_cap_data))

D = dict() 
for it in large_cap_data: 
    D[it['symbol']] = True
    #print(it['symbol'])

pickle.dump(D, open('data/large_cap_filter_dict.pkl', 'wb'))
