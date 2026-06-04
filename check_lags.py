import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()

db = MongoClient(os.getenv('MONGO_URI'))['aqi_predictor']
records = list(db['aqi_features'].find({}, {'_id': 0}))
df = pd.DataFrame(records).sort_values('timestamp').reset_index(drop=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print('=== LAG INTEGRITY CHECK ===')
df['expected_lag1'] = df['aqi'].shift(1)
df['lag1_error'] = (df['aqi_lag_1h'] - df['expected_lag1']).abs()
print(f'aqi_lag_1h matches shift(1): {(df.lag1_error < 0.01).mean()*100:.1f}% of rows')
print(f'Mean lag1 error:             {df.lag1_error.mean():.4f}')
print(f'Rows where lag1 is wrong:    {(df.lag1_error > 1).sum()}')

print()
df['expected_rolling3'] = df['aqi'].shift(1).rolling(3, min_periods=1).mean()
df['rolling3_error'] = (df['aqi_rolling_3h'] - df['expected_rolling3']).abs()
print(f'aqi_rolling_3h correct: {(df.rolling3_error < 0.1).mean()*100:.1f}% of rows')
print(f'Rows where rolling3 is wrong: {(df.rolling3_error > 1).sum()}')

print()
df['gap'] = df['timestamp'].diff().dt.total_seconds() / 3600
big_gaps = df[df['gap'] > 1.5]
print(f'Timestamp gaps > 1.5h: {len(big_gaps)}')
if len(big_gaps) > 0:
    print(big_gaps[['timestamp', 'gap', 'aqi', 'aqi_lag_1h']].head(10).to_string())

print()
print('=== SAMPLE ROWS (lag alignment check) ===')
print(df[['timestamp', 'aqi', 'aqi_lag_1h', 'aqi_lag_2h', 'aqi_rolling_3h']].iloc[10:16].to_string())

print()
print('=== NAIVE vs MODEL RMSE ===')
naive_rmse = np.sqrt(((df['aqi'] - df['aqi_lag_1h'])**2).mean())
print(f'Naive predictor (just use lag_1h): RMSE = {naive_rmse:.4f}')
print(f'Your model RMSE:                   8.31')
print(f'Gap (model should be LOWER):       {8.31 - naive_rmse:.4f}')