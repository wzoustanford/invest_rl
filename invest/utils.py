import requests, pdb, json, os, re, copy, time, pickle
from datetime import datetime, timedelta
from ts_data_struct import BiHashList
from openai import OpenAI

FINANCIAL_KEY = "1e347f859bc1eaa56334ad8c5dc10924" #"897d694a6ab563cb079534513ee9ea1a"#"1e347f859bc1eaa56334ad8c5dc10924"
OPENAI_KEY = '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'#"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"

def aggregate_tickers_RL(data_file_list, start_idx, end_idx_plus1, exp_id): 
    """
    Function to aggregate the stock tickers into a unified hash across a time frame 
    """
    import sys
    
    save_pkl_name = exp_id + '_ticker_hash.pkl' 
    cnt = 0
    first_D = dict() 
    final_D = dict() 
    
    total_files = end_idx_plus1 - start_idx
    print(f"Processing {total_files} files for ticker hash creation...")
    print(f"Files range: {start_idx} to {end_idx_plus1-1}")
    sys.stdout.flush()
    
    for file_num, idx in enumerate(range(start_idx, end_idx_plus1)): 
        filename = data_file_list[idx] 
        
        # Progress logging every 10 files
        if file_num % 10 == 0 or file_num == total_files - 1:
            print(f"Processing file {file_num+1}/{total_files} (idx={idx}): {filename}")
            sys.stdout.flush()
        
        try:
            f = open(filename, 'rb') 
            D = pickle.load(f) 
            f.close()
            
            test_features = D['trainFeature'] 
            all_train_tickers = D['all_train_tickers'] 
            series = D['train_in_portfolio_series']
            
            if idx == start_idx: 
                for ticker_idx, ticker in enumerate(all_train_tickers): 
                    first_D[ticker] = True 
                print(f"Initial file contains {len(all_train_tickers)} tickers") 
                print(f"Sample tickers: {all_train_tickers[:5]}...")
                sys.stdout.flush()
            else: 
                cur_D = dict() 
                for ticker_idx, ticker in enumerate(all_train_tickers): 
                    if series[ticker_idx, 0] != 0: 
                        cur_D[ticker] = True 
                
                # Filter tickers that don't appear in current file
                K = list(first_D.keys())
                removed_count = 0
                for first_ticker in K: 
                    if first_ticker not in cur_D: 
                        del first_D[first_ticker] 
                        removed_count += 1
                
                if file_num % 20 == 0:  # Log every 20 files
                    print(f"  After file {file_num+1}: {len(first_D)} tickers remaining (removed {removed_count})")
                    sys.stdout.flush()
                    
        except Exception as e:
            print(f"ERROR processing file {file_num+1} ({filename}): {e}")
            sys.stdout.flush()
            continue
    
    print(f"Ticker filtering complete. {len(first_D)} tickers found across all files.")
    print("Applying market cap filter...")
    sys.stdout.flush()
    
    ## finally filter by 20 bn market cap 
    try:
        D_20bn = pickle.load(open('/home/ubuntu/code/angle_rl/invest/data/large_cap_filter_dict.pkl', 'rb')) 
        print(f"Market cap filter loaded with {len(D_20bn)} large cap tickers")
    except Exception as e:
        print(f"ERROR loading market cap filter: {e}")
        D_20bn = {}
    
    for t in first_D.keys(): 
        if t in D_20bn: 
            final_D[t] = cnt 
            cnt += 1 

    print(f'Final total number of tickers after market cap filter: {cnt}') 
    print(f"Sample final tickers: {list(final_D.keys())[:10]}...")
    
    saveD = dict() 
    saveD['hash_D'] = final_D 
    saveD['num_tickers'] = cnt 
    
    print(f"Saving ticker hash to: {save_pkl_name}")
    pickle.dump(saveD, open(save_pkl_name, 'wb'))
    print("Ticker hash creation completed!")
    sys.stdout.flush() 

def load_data_list(data_list_filename, return_count=False, load_as_pickle_dict=False, f_num=None): 
    data_list = [] 
    cnt = 0 
    f = open(data_list_filename, 'r') 
    l = f.readline() 
    while l:
        cnt += 1
        if f_num is not None and cnt >= f_num: 
            break
        if load_as_pickle_dict:
            data_list.append(pickle.load(open(l.strip(), 'rb')))
        else: 
            data_list.append(l.strip()) 
        l = f.readline() 
    f.close() 
    if return_count is True: 
        return data_list, cnt 
    else: 
        return data_list 

def find_file_in_dir(dir, reg_pattern):
    files_and_dirs = os.listdir(dir)
    regex = re.compile(reg_pattern)
    res = []
    for file in files_and_dirs:
        if regex.match(file):
            res.append(file)
    return res 

def read_json_file(file_path):
    """
    Reads a JSON file and returns its content.

    Args:
        file_path: The path to the JSON file.

    Returns:
        The content of the JSON file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def find_closest_datetime_condition(datetime_list, target_datetime, condition):
    """
    Finds the closest datetime to a target datetime in a list.

    Args:
        datetime_list: A list of datetime objects.
        target_datetime: The datetime object to find the closest to.

    Returns:
        The closest datetime object in the list to the target datetime, or None if the list is empty.
    """
    """
    # Example usage:
    dates = [
        datetime(2025, 3, 20),
        datetime(2025, 3, 22),
        datetime(2025, 3, 25),
        datetime(2025, 3, 28),
    ]
    target_date = datetime(2025, 3, 23)

    closest_date = find_closest_datetime(dates, target_date)
    print(f"The closest date to {target_date} is {closest_date}") # Output: The closest date to 2025-03-23 00:00:00 is 2025-03-22 00:00:00"
    """

    if not datetime_list:
        return None
    
    if condition == 'strictly_after':
        after_datetime_list = []
        for dt in datetime_list: 
            if dt > target_datetime: #strictly larger, so if a datetime in the list is given, look for next datetime
                after_datetime_list.append(dt)
        res = min(after_datetime_list, key=lambda x: abs(x - target_datetime))
    elif condition == 'at_or_after':
        after_datetime_list = []
        for dt in datetime_list: 
            if dt >= target_datetime: #at or after, greater or equal 
                after_datetime_list.append(dt)
        res = min(after_datetime_list, key=lambda x: abs(x - target_datetime))
    elif condition == 'strictly_before': 
        before_datetime_list = []
        for dt in datetime_list: 
            if dt < target_datetime: #strictly larger, so if a datetime in the list is given, look for next datetime
                before_datetime_list.append(dt)
        res = min(before_datetime_list, key=lambda x: abs(x - target_datetime))
    elif condition == 'at_or_after_prefered': 
        after_datetime_list = []
        for dt in datetime_list: 
            if dt >= target_datetime: #at or after, greater or equal 
                after_datetime_list.append(dt)
        if len(after_datetime_list) > 0: 
            res = min(after_datetime_list, key=lambda x: abs(x - target_datetime))
        else: 
            res = min(datetime_list, key=lambda x: abs(x - target_datetime))    
    else: 
        res = min(datetime_list, key=lambda x: abs(x - target_datetime))

    return res 

def find_closest_datetime(datetime_list, target_datetime):
    """
    Finds the closest datetime to a target datetime in a list.

    Args:
        datetime_list: A list of datetime objects.
        target_datetime: The datetime object to find the closest to.

    Returns:
        The closest datetime object in the list to the target datetime, or None if the list is empty.
    """
    """
    # Example usage:
    dates = [
        datetime(2025, 3, 20),
        datetime(2025, 3, 22),
        datetime(2025, 3, 25),
        datetime(2025, 3, 28),
    ]
    target_date = datetime(2025, 3, 23)

    closest_date = find_closest_datetime(dates, target_date)
    print(f"The closest date to {target_date} is {closest_date}") # Output: The closest date to 2025-03-23 00:00:00 is 2025-03-22 00:00:00"
    """

    if not datetime_list:
        return None

    return min(datetime_list, key=lambda x: abs(x - target_datetime))

def get_finance_api_data(url, max_retries=3, wait_time=5, key_only=False):
    retries = 0

    while retries < max_retries:
        try:
            headers = {"User-Agent": "Mozilla/5.0"}
            if key_only is True: 
                response = requests.get(f"{url}?apikey={FINANCIAL_KEY}",headers=headers)
            else: 
                response = requests.get(f"{url}&apikey={FINANCIAL_KEY}",headers=headers)

            response.raise_for_status()
            return response.json()

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit hit! Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                retries += 1
            else:
                print(f"HTTP Error {e.response.status_code}: {e.response.reason}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            break

        except Exception as e:
            print(f"Error: {e}")
            break

    return None

def build_price_volume_chart_data(stock_list, start_date, end_date, url_str):
    D = {}
    count = 0
    full_count = len(stock_list)
    for item in stock_list:
        symbol = item['symbol']
        count += 1
        print(f'{count}/{str(full_count)} Requesting price/volume data for {url_str} chart from {start_date} to {end_date} for {symbol} ...')
        if '15min' in url_str:
            res = query_fmp_15min_monthly_and_append_res(start_date, end_date, url_str, symbol)
        else: 
            url = f'https://financialmodelingprep.com/stable/{url_str}?symbol={symbol}&from={start_date}&to={end_date}'
            res = get_finance_api_data(url=url)
        if not res or len(res) == 0:
            print(f"Failed to get data for {symbol}")
            continue
        res.reverse()
        prices = BiHashList()
        volumes = BiHashList()
        for item in res:
            price_key = 'price' if 'light' in url_str else 'close'
            prices.append(item['date'], item[price_key])
            volumes.append(item['date'], item['volume'])
        D[symbol] = {'prices':prices, 'volumes':volumes}
    return D

def query_fmp_15min_monthly_and_append_res(start_date, end_date, url_str, symbol):
    fmt="%Y-%m-%d"
    res = []
    ED = datetime.strptime(end_date, fmt)
    SD = ED - timedelta(days=29)
    while SD >= datetime.strptime(start_date, fmt): 
        url = f'https://financialmodelingprep.com/stable/{url_str}?symbol={symbol}&from={SD.strftime(fmt)}&to={ED.strftime(fmt)}'
        r = get_finance_api_data(url=url) 
        if not r: 
            print(f"Failed to get monthly 15min data for {symbol}")
            return res 
        res += r 
        ED -= timedelta(days=30)
        SD -= timedelta(days=30)
    return res 

def get_news_full_string_ticker(tickers, from_date, to_date, page_limit=1):
    text_str = ""
    for page in range(page_limit):
        url = f"https://financialmodelingprep.com/api/v3/stock_news?tickers={tickers}&page={page+1}&from={from_date}&to={to_date}"
        res = get_finance_api_data(url)
        additional_str = ""
        if res: 
            for i in range(len(res) - 1, -1, -1):
                r = res[i]
                #print(r["publishedDate"])
                #print(len(r["text"].split(" ")))
                if r["publishedDate"] is not None: 
                    additional_str += " published date:"+r["publishedDate"]
                if r["title"] is not None:
                    additional_str += " title: "+r["title"]
                if r["text"] is not None:
                    additional_str += " text: "+r["text"]
            text_str += additional_str 
    return text_str

#def get_openai_embedding(url, ): 

def get_openai_embedding(text_str, max_retries=3, wait_time=5):
    client = OpenAI(api_key=OPENAI_KEY)
    max_char_len = 30000 #49152
    if len(text_str) > max_char_len:
        text_str = copy.deepcopy(text_str[:max_char_len])
    retries = 0
    while retries < max_retries:
        try:
            response = client.embeddings.create(
                input=text_str,
                model="text-embedding-3-large"
            )
            emb = response.data[0].embedding
            return emb 

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"Rate limit hit! Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                wait_time *= 2
                retries += 1
            else:
                print(f"HTTP Error {e.response.status_code}: {e.response.reason}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            break

        except Exception as e:
            print(f"Error: {e}")
            break
    default_vec = [0.0 for i in range(3072)]
    return default_vec

def get_news_embedding(tickers, from_date, to_date, page_limit=1):
    print(f"getting news embedding for ticker {tickers}, from date: {from_date}, to date: {to_date}")
    text_str = get_news_full_string_ticker(tickers, from_date, to_date, page_limit=page_limit)
    res = get_openai_embedding(text_str)
    return res, text_str

