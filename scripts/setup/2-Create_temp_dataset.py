import xarray as xr
import pandas as pd
import numpy as np
import os
from pyproj import Transformer

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', '..','data' ,'base_data', 'era_downloads')
data_dir = os.path.abspath(data_dir)

merged_nc     = os.path.join(data_dir, 'era5land_merged.nc')

excel_file = os.path.join(script_dir, '..', '..','data' ,'base_data', 'farms with coordinates.xlsx')
excel_file = os.path.abspath(excel_file)

output_csv = os.path.join(script_dir, '..', '..','data' ,'temp_results', 'farms_daily_t2m_2021-2025.csv')
output_csv = os.path.abspath(output_csv)

# UTM zone assumption for transformed coords (adjust if needed)
utm_crs       = 25831  # ETRS89 / UTM zone 31N
wgs84_crs     = 4326

# Variable name in the dataset
var_name      = 't2m'

# Load merged ERA5-Land daily data
print("Loading merged ERA5-Land dataset…")
ds = xr.open_dataset(merged_nc)
if var_name not in ds.data_vars:
    raise KeyError(f"Variable '{var_name}' not found in dataset: {list(ds.data_vars)}")

# Identify time/lat/lon dims
coords = ds[var_name].coords
time_dim = next(d for d in coords if 'time' in d.lower())
lat_dim  = next(d for d in coords if 'lat'  in d.lower())
lon_dim  = next(d for d in coords if 'lon'  in d.lower())

# Extract coordinate axes
lats = ds[lat_dim].values
lons = ds[lon_dim].values

# Read farm list and prepare coordinates 
print("Reading farm Excel…")
df = pd.read_excel(
    excel_file,
    usecols=['MO','COORDENADA X EXPLOTACIÓ','COORDENADA Y EXPLOTACIÓ']
).rename(
    columns={
        'COORDENADA X EXPLOTACIÓ':'x','COORDENADA Y EXPLOTACIÓ':'y'
    }
)

# Determine if coords appear to be lat/lon or UTM
# Simple heuristic: if values sign or range outside [ -180,180 ], treat as UTM
if df['x'].abs().max() > 180 or df['y'].abs().max() > 90:
    print("Detected UTM coordinates - transforming to WGS84 lat/lon…")
    transformer = Transformer.from_crs(utm_crs, wgs84_crs, always_xy=True)
    df[['lon','lat']] = df.apply(
        lambda row: transformer.transform(row['x'], row['y']),
        axis=1, result_type='expand'
    )
else:
    print("Detected lon/lat coordinates…")
    df = df.rename(columns={'x':'lon','y':'lat'})

# Extract daily temperature for each farm 
results = []
for idx, row in df.iterrows():
    farm = row['MO']
    lon0, lat0 = row['lon'], row['lat']
    print(f"Processing farm {farm}: target lon={lon0:.4f}, lat={lat0:.4f}…")

    # Find nearest grid indices manually for clarity
    i_lat = int(np.abs(lats - lat0).argmin())
    i_lon = int(np.abs(lons - lon0).argmin())
    lat_sel = float(lats[i_lat])
    lon_sel = float(lons[i_lon])
    print(f"  Nearest grid point -> lon={lon_sel:.4f}, lat={lat_sel:.4f}")

    # Select the point time series
    da_pt = ds[var_name].isel({lat_dim:i_lat, lon_dim:i_lon})
    # Convert Kelvin to Celsius
    da_c = da_pt - 273.15

    # Build DataFrame
    df_ts = da_c.to_series().reset_index()
    df_ts = df_ts.rename(columns={time_dim:'Data_censo', var_name:'t2m_C'})
    df_ts['MO'] = farm

    # Check missing
    n_tot = len(df_ts)
    n_nan = df_ts['t2m_C'].isna().sum()
    print(f"  Days total={n_tot}, missing={n_nan}")

    # Keep only valid values
    df_valid = df_ts.dropna(subset=['t2m_C'])
    results.append(df_valid[['MO','Data_censo','t2m_C']])

# Concatenate and save
if results:
    df_final = pd.concat(results, ignore_index=True)
    print(f"Writing CSV with {len(df_final)} records to {output_csv}")
    df_final.to_csv(output_csv, index=False)
else:
    print("No valid data extracted for any farm.")

print("All done.")
