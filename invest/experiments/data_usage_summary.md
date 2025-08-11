# Data Files and Time Period Summary for LSTM vs Sequential Comparison

## Dataset Overview
- **Total files available**: 1,070 files
- **File format**: Each file contains both training data (360 days) and test data (25 days holding period)
- **File naming**: `model_data_single_step_trainingtimelength360d_buyselltimelength25d_training_data_start_date_YYYY_MM_DD_test_data_start_date_YYYY_MM_DD_newsFeaturesFalse_alpacafracfiltered.pkl`

## Files Used for Comparison Experiment

### Training Period
- **Files used**: Files #605-704 (100 files total)
- **Training data dates**: Approximately June 2022 - December 2022
- **Purpose**: Used to train both LSTM and Sequential Supervised models before testing

### Testing Period
- **Files used**: Files #705-1070 (366 files total)
- **Test data dates**: January 1, 2023 - May 5, 2024
- **Total duration**: ~16 months of daily data

### Specific Date Ranges

#### 2023 Coverage
- **Start**: January 1, 2023 (file #705)
  - File: `test_data_start_date_2023_01_01`
- **End**: December 31, 2023
- **Months covered**: All 12 months of 2023

#### 2024 Coverage  
- **Start**: January 1, 2024
- **End**: May 5, 2024 (file #1070)
  - File: `test_data_start_date_2024_05_05`
- **Months covered**: January through early May 2024 (approximately 4 months)

## Monthly Breakdown in Experiment

The experiment aggregated daily files into monthly periods:
- Each "month" in the results represents approximately 20 trading days
- **19 monthly periods** were tested in total:
  - Months 1-12: Covering 2023 (Jan-Dec)
  - Months 13-19: Covering 2024 (Jan-May)

### Monthly Aggregation Method
- **Month size**: 20 files (approximately 20 trading days)
- **Processing**: Each month's files were processed sequentially
- **Returns calculation**: Monthly returns were averaged across all trades within each month period

## Data Structure Details

Each pickle file contains:
- **trainFeature**: Features for 360-day training period (varying dimensions ~247-249 features, ~7500+ stocks)
- **train_in_portfolio_series**: Price series for training period
- **testFeature**: Features for 25-day test/trading period
- **test_in_portfolio_series**: Price series for test period (used for calculating actual returns)

## Key Observations
1. The data files represent consecutive trading days with overlapping training windows
2. Each file's test period represents a potential 25-day holding period starting from the test_data_start_date
3. The sequential nature of files allows for backtesting strategies across continuous time periods
4. Files have varying numbers of stocks (7500-7600) and features (247-249), requiring normalization