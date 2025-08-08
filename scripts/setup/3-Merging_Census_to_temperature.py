import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
census_csv = os.path.join(script_dir, '..', '..','data' ,'base_data', 'census_all.csv')
census_csv = os.path.abspath(census_csv)

temp_csv = os.path.join(script_dir, '..', '..','data' ,'temp_results', 'farms_daily_t2m_2021-2025.csv')
temp_csv = os.path.abspath(temp_csv)

output_csv = os.path.join(script_dir, '..', '..','data' ,'temp_results', 'census_all_t2m.csv')
output_csv = os.path.abspath(output_csv)

# Load census dataset
print(f"Loading census data from {census_csv} (sep=';')")
census = pd.read_csv(
    census_csv,
    sep=';',                        # semicolon delimiter
    parse_dates=['Data_censo']
)

# Load temperature dataset
print(f"Loading temperature data from {temp_csv}")
temp = pd.read_csv(temp_csv)

# Ensure the date column is present and parsed
if 'Data_censo' not in temp.columns:
    # try lowercase or alternate names
    alt = [c for c in temp.columns if c.strip().lower() == 'data_censo' or 'date' in c.lower()]
    if len(alt) == 1:
        temp = temp.rename(columns={alt[0]: 'Data_censo'})
        print(f"Renamed column '{alt[0]}' to 'Data_censo'")
    else:
        raise ValueError(f"Date column 'Data_censo' not found in temperature file. Available columns: {list(temp.columns)}")
temp['Data_censo'] = pd.to_datetime(temp['Data_censo'])

# Merge into a single dataset
print("Merging datasets on ['MO', 'Data_censo'] (inner join)…")
merged = pd.merge(
    census,
    temp[['MO', 'Data_censo', 't2m_C']],
    on=['MO', 'Data_censo'],
    how='inner'
)

# Report dropped rows
n_census = len(census)
n_merged = len(merged)
print(f"Census rows: {n_census}, after merge: {n_merged}, dropped: {n_census - n_merged}")

# Check for missing temperature values per farm 
print("Checking for missing temperature values per farm…")
missing_per_farm = merged.groupby('MO')['t2m_C'].apply(lambda s: s.isna().sum())
for farm_id, n_missing in missing_per_farm.items():
    print(f"  Farm {farm_id}: missing t2m_C values = {n_missing}")

# Export merged CSV
print(f"Writing merged dataset to {output_csv}")
merged.to_csv(output_csv, index=False)
print("Done.")
