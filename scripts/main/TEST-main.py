# In[1]: Importing necessary libraries
import functions as f
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

# In[2]: Reading the dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', '..','data' ,'temp_results','census_all_t2m.csv')
csv_path = os.path.abspath(csv_path)

data = pd.read_csv(
    csv_path,
    dtype={"MO": str}
)
all_results = pd.DataFrame(columns=['MO', 'Insertion date', 'Total water consumption'])

# In[3]: Formatting dataset in English and converting to correct data types
formatted_data = f.formatting_dataset(data)

# In[4]: Prepare for loop
farm_ids = formatted_data['MO'].unique()
farm_ids = ["0230GI", "0030CY", "0040EB", "0040KK"]
total    = len(farm_ids)

# collect farms with missing temperature
missing_logs = []

# In[5]: Loop over farms, skipping those with missing t2m_C
for idx, farm_id in enumerate(farm_ids, start=1):
    print(f"Processing {idx}/{total} – Farm ID: {farm_id}")
    
    farm_data     = formatted_data[formatted_data['MO'] == farm_id]
    missing_count = farm_data['t2m_C'].isna().sum()
    if missing_count > 0:
        missing_logs.append({'MO': farm_id, 'missing_t2m_C': missing_count})
        continue
    
    farm_results = f.farm_prediction(formatted_data, farm_id=farm_id)
    if farm_results is not None:
        all_results = pd.concat([all_results, farm_results], ignore_index=True)

if missing_logs:
    missing_df = pd.DataFrame(missing_logs)
    missing_log_path = os.path.join(script_dir, '..', '..','data' ,'final_results','farms_missing_t2m.csv')
    missing_log_path = os.path.abspath(missing_log_path)
    missing_df.to_csv(
        missing_log_path,
        index=False
    )
    print(f"Skipped {len(missing_logs)} farms due to missing t2m_C; details in farms_missing_t2m.csv")

# In[6]: Save all_results
if not all_results.empty:
    all_results_path = os.path.join(script_dir, '..', '..','data' ,'temp_results','results.csv')
    all_results_path = os.path.abspath(all_results_path)
    all_results.to_csv(
        all_results_path,
        index=False
    )
else:
    print("No data to save.")

# In[7]: Merging results with the original data
all_results.rename(columns={'Insertion date': 'Data_censo'}, inplace=True)
data['Data_censo'] = pd.to_datetime(data['Data_censo'], errors='coerce')

merged = data.merge(
    all_results,
    on=['MO', 'Data_censo'],
    how='left'
)
merged_census_path = os.path.join(script_dir, '..', '..','data' ,'final_results','test-complete_census_all.csv')
merged_census_path = os.path.abspath(merged_census_path)
merged.to_csv(
    merged_census_path,
    index=False
)
