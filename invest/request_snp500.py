import pickle, pdb
from utils import get_finance_api_data 

url = 'https://financialmodelingprep.com/api/v3/sp500_constituent'
snp500_list = get_finance_api_data(url=url, key_only=True)

pdb.set_trace() 

Dsnp500 = dict() 
for it in snp500_list: 
    Dsnp500[it['symbol']] = True

pickle.dump({"Dsnp500":Dsnp500}, open('data/snp500_dict.pkl', 'wb'))
