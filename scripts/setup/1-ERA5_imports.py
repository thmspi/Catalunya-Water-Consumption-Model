#In[1]: Importing libraries

import cdsapi
import xarray as xr
import os

#In[2]: Variables and dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..','data' ,'base_data', 'era_downloads')
data_dir = os.path.abspath(data_dir)

c = cdsapi.Client()

start_year = 2021
end_year = 2025
end_month = 5

# Catalonia region
area = [42.0, -1.0, 38.0, 5.0]

# Year range [X;Y[
years = list(range(start_year, end_year+1)) 

#In[3]: fetching data

for year in years:
    # as we don't go till end of 2025
    if year < end_year:
        months = [f"{m:02d}" for m in range(1, 13)]
    else:
        months = [f"{m:02d}" for m in range(1, end_month+1)]  
    # Always request days 01–31 (ERA5 will drop non-existent dates)
    days = [f"{d:02d}" for d in range(1, 32)]

    target_file = os.path.join(data_dir, f"era5land_cat_{year}.nc")
    print(f"Requesting {year}, saving to {target_file}...")
    c.retrieve(
        'derived-era5-land-daily-statistics',
        {
            'variable':        ['2m_temperature'],      # Mean 2 m temperature
            'daily_statistic': 'daily_mean',            # Compute daily mean
            'time_zone':       'utc+00:00',             # Boundaries in UTC
            'frequency':       '6_hourly',              # 6-hourly data for stats
            'area':            area,
            'year':            [str(year)],
            'month':           months,
            'day':             days,
            'format':          'netcdf',
        },
        target_file
    )

#In[4]: Merging into one daily dataset and exporting it

# --- Verify downloads before merging ---
file_list = [os.path.join(data_dir, f"era5land_cat_{y}.nc") for y in years]
missing = [f for f in file_list if not os.path.exists(f)]
if missing:
    raise FileNotFoundError(f"Missing files, cannot merge: {missing}")

# --- Merge yearly files into a single dataset ---
print("Merging yearly files into a single dataset...")
ds = xr.open_mfdataset(file_list, combine='by_coords')
print("Dataset dimensions:", ds.dims)

# Identify the time dimension dynamically
_time_dim = next((d for d in ds.dims if 'time' in d.lower()), None)
if _time_dim is None:
    raise ValueError(f"No time dimension found. Available dims: {list(ds.dims)}")
print(f"Using '{_time_dim}' as time dimension for resampling")

# Resample to daily mean (if needed) and merge (ensure consistency as it should already be a daily mean)
ds_daily = ds.resample({ _time_dim: '1D' }).mean(dim=_time_dim)

# Save the merged dataset
merged_file = os.path.join(
    data_dir,
    f"era5land_merged.nc"
)
ds_daily.to_netcdf(merged_file)
print(f"Merged dataset saved to {merged_file}")

# %%
