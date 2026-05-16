## set-up script for raptor machines 

mkdir code 
cd code 

git clone https://wzoustanford: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@github.com/wzoustanford/angle_rl.git

mkdir ~/.aws/
touch ~/.aws/credentials 
echo -e "[default]\naws_access_key_id = AKIASDQ36EOJONW2RDHZ\naws_secret_access_key = xxx" > ~/.aws/credentials
cd angle_rl/invest/data 

#aws s3 cp --recursive s3://illumenti-backend-general/angle_rl_data/dec23_to_mar24_benchmark_data/ ./
aws s3 cp --recursive s3://illumenti-backend-general/angle_rl_data/price_data/ ./

cd ~/

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install bidict
pip3 install openai
