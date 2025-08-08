import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', '..','data', 'temp_results', 'farms_daily_t2m_2021-2025.csv')
csv_path = os.path.abspath(csv_path)
df = pd.read_csv(csv_path)

nan_rows = df[df['t2m_C'].isna()]

if nan_rows.empty:
    print("No NaN values found in 't2m_C'")
else:
    print("Rows with NaN values in 't2m_C'")
    print(nan_rows)
