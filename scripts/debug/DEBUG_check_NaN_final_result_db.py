import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(script_dir, '..', '..','data', 'final_results', 'complete_census_all.csv')
csv_path = os.path.abspath(csv_path)
df = pd.read_csv(csv_path)

nan_rows = df[df['t2m_C'].isna() | df['Total water consumption'].isna()]

if nan_rows.empty:
    print("No NaN values found in 't2m_C' or 'Total water consumption'.")
else:
    print("Rows with NaN values in 't2m_C' or 'Total water consumption':")
    print(nan_rows)
