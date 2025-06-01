## set-up script for raptor machines 

mkdir code 
cd code 

git clone https://wzoustanford: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@github.com/wzoustanford/angle_rl.git

cd angle_rl/invest/data 

aws s3 cp --recursive s3://illumenti-backend-general/angle_rl_data/dec23_to_mar24_benchmark_data/ ./
aws s3 cp --recursive s3://illumenti-backend-general/angle_rl_data/price_data/ ./
