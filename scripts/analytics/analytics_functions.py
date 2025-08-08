import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_farm_data(df: pd.DataFrame, farm_id: str):

    # verify that we are working with date type
    if not pd.api.types.is_datetime64_any_dtype(df['Data_censo']):
        df['Data_censo'] = pd.to_datetime(df['Data_censo'])

    # create subset for the specified farm
    df_farm = df[df['MO'] == farm_id].copy()
    if df_farm.empty:
        print(f"No data found for farm {farm_id}")
        return

    # Sort by date
    df_farm.sort_values('Data_censo', inplace=True)

    # 1) Plot total water consumption
    plt.figure(figsize=(10, 5))
    plt.plot(df_farm['Data_censo'], df_farm['Total water consumption'],
             marker='o', linestyle='-')
    plt.title(f"Total Water Consumption for {farm_id}")
    plt.xlabel('Date')
    plt.ylabel('Total water consumption')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Plot total weight evolution
    plt.figure(figsize=(10, 5))
    plt.plot(df_farm['Data_censo'], df_farm['total_peso'],
             marker='o', linestyle='-')
    plt.title(f"Total Weight Evolution for {farm_id}")
    plt.xlabel('Date')
    plt.ylabel('Total weight')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Plot number of animals evolution
    plt.figure(figsize=(10, 5))
    plt.plot(df_farm['Data_censo'], df_farm['num_anim_total'],
             marker='o', linestyle='-')
    plt.title(f"Number of Animals Evolution for {farm_id}")
    plt.xlabel('Date')
    plt.ylabel('Number of animals')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Calculate and print total consumed water
    total_water = df_farm['Total water consumption'].sum()
    print(f"Total consumed water for {farm_id}: {total_water}L")


def plot_farm_data(df: pd.DataFrame, farm_id: str):

    if 'Data_censo' not in df.columns:
        raise KeyError("DataFrame must contain 'Data_censo' column")
    if not pd.api.types.is_datetime64_any_dtype(df['Data_censo']):
        df = df.copy()
        df['Data_censo'] = pd.to_datetime(df['Data_censo'])

    # create subset for the specified farm
    df_farm = df[df['MO'] == farm_id].copy()
    if df_farm.empty:
        print(f"No data found for farm {farm_id}")
        return

    # Sort by date
    df_farm.sort_values('Data_censo', inplace=True)

    # 1) Plot total water consumption
    plt.figure(figsize=(10, 5))
    plt.plot(df_farm['Data_censo'], df_farm['Total water consumption'],
             marker='o', linestyle='-')
    plt.title(f"Total Water Consumption for {farm_id}")
    plt.xlabel('Date')
    plt.ylabel('Total water consumption')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Plot total weight evolution
    plt.figure(figsize=(10, 5))
    plt.plot(df_farm['Data_censo'], df_farm['total_peso'],
             marker='o', linestyle='-')
    plt.title(f"Total Weight Evolution for {farm_id}")
    plt.xlabel('Date')
    plt.ylabel('Total weight')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Plot number of animals evolution
    plt.figure(figsize=(10, 5))
    plt.plot(df_farm['Data_censo'], df_farm['num_anim_total'],
             marker='o', linestyle='-')
    plt.title(f"Number of Animals Evolution for {farm_id}")
    plt.xlabel('Date')
    plt.ylabel('Number of animals')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Calculate and print total consumed water
    total_water = df_farm['Total water consumption'].sum()
    print(f"Total consumed water for {farm_id}: {total_water:.2e} L")

def merge_coordinates(main_df: pd.DataFrame, coords_df: pd.DataFrame,
                      coord_y_col: str = 'COORDENADA Y EXPLOTACIÓ',
                      coord_x_col: str = 'COORDENADA X EXPLOTACIÓ') -> pd.DataFrame:
    required = ['MO', coord_y_col, coord_x_col]
    for col in required:
        if col not in coords_df.columns:
            raise KeyError(f"coords_df must contain '{col}' column")
    if 'MO' not in main_df.columns:
        raise KeyError("main_df must contain 'MO' column")

    merged_df = main_df.merge(
        coords_df[['MO', coord_y_col, coord_x_col]],
        on='MO', how='left'
    )

    cols = ['MO'] + [c for c in merged_df.columns if c != 'MO']
    merged_df = merged_df[cols]

    return merged_df

def analyze_water_by_location(df: pd.DataFrame,
                              coord_y_col: str = 'COORDENADA Y EXPLOTACIÓ',
                              coord_x_col: str = 'COORDENADA X EXPLOTACIÓ',
                              num_anim_col: str = 'num_anim_total',
                              bins: list = None) -> None:
    if bins is None:
        bins = [0, 500, 1000, 2000, 3000, float('inf')]
    labels = ['<500', '500-1000', '1000-2000', '2000-3000', '>3000']
    df['size_group'] = pd.cut(df[num_anim_col], bins=bins, labels=labels)
    grouped = df.groupby(['MO','size_group'], observed=False).agg({
        'Total water consumption': 'mean',
        coord_x_col: 'first',
        coord_y_col: 'first'
    }).reset_index()
    for label in labels:
        subset = grouped[grouped['size_group']==label]
        x = subset[coord_x_col]
        y = subset[coord_y_col]
        c = subset['Total water consumption']
        plt.figure(figsize=(8,6))
        plt.hexbin(x, y, C=c, gridsize=50, reduce_C_function=np.mean, cmap='viridis')
        plt.colorbar(label='Avg water consumption')
        plt.xlabel(coord_x_col)
        plt.ylabel(coord_y_col)
        plt.title(f'Water consumption for size {label}')
        plt.tight_layout()
        plt.show()

def analyze_geographical_correlation(result : pd.DataFrame, coordinates : pd.DataFrame):
    merged = merge_coordinates(result, coordinates)
    analyze_water_by_location(merged)

def merge_coordinates(main_df: pd.DataFrame, coords_df: pd.DataFrame,
                      coord_y_col: str = 'COORDENADA Y EXPLOTACIÓ',
                      coord_x_col: str = 'COORDENADA X EXPLOTACIÓ') -> pd.DataFrame:
    
    # Check all columns are present
    required = ['MO', coord_y_col, coord_x_col]
    for col in required:
        if col not in coords_df.columns:
            raise KeyError(f"coords_df must contain '{col}' column")
    if 'MO' not in main_df.columns:
        raise KeyError("main_df must contain 'MO' column")
    merged_df = main_df.merge(coords_df[['MO', coord_y_col, coord_x_col]],
                              on='MO', how='left')
    cols = ['MO'] + [c for c in merged_df.columns if c != 'MO']
    return merged_df[cols]


def analyze_water_by_location(df: pd.DataFrame,
                              coord_y_col: str = 'COORDENADA Y EXPLOTACIÓ',
                              coord_x_col: str = 'COORDENADA X EXPLOTACIÓ',
                              num_anim_col: str = 'num_anim_total',
                              bins: list = None) -> None:
    if bins is None:
        bins = [0, 500, 1000, 2000, 3000, float('inf')]
    labels = ['C1 : <500', 'C2 : 500-1000', 'C3 : 1000-2000', 'C4 : 2000-3000', 'C5 : >3000']
    df['size_group'] = pd.cut(df[num_anim_col], bins=bins, labels=labels)
    grouped = df.groupby(['MO','size_group'], observed=False).agg({
        'Total water consumption': 'mean',
        coord_x_col: 'first',
        coord_y_col: 'first'
    }).reset_index()

    # Plot each maps for each group
    for label in labels:
        subset = grouped[grouped['size_group']==label]
        x = subset[coord_x_col]
        y = subset[coord_y_col]
        c = subset['Total water consumption']
        plt.figure(figsize=(8,6))
        plt.hexbin(x, y, C=c, gridsize=50, reduce_C_function=np.mean, cmap='viridis')
        plt.colorbar(label='Avg water consumption')
        plt.xlabel(coord_x_col)
        plt.ylabel(coord_y_col)
        plt.title(f'Water consumption for size {label}')
        plt.tight_layout()
        plt.show()

    df["Data_censo"] = pd.to_datetime(df["Data_censo"])
    df['year'] = df["Data_censo"].dt.year

    annual = (
        df
        .groupby(['year', 'size_group'], observed=False)["Total water consumption"]
        .mean()
        .unstack('size_group')
        .reindex(columns=labels)
    )

    if annual.empty:
        print("No annual data to plot.")
        return

    plt.figure(figsize=(10, 6))
    for label in labels:
        if label in annual.columns:
            plt.plot(annual.index, annual[label], marker='o', label=label)

    plt.xlabel('Year')
    plt.ylabel("Average water consumption per day")
    plt.title('Yearly Average Evolution of Water Consumption by Farm Size Group')
    plt.legend(title='Size Group')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    annual = (
        df
        .groupby(['year', 'size_group'], observed=False)["Total water consumption"]
        .sum()
        .unstack('size_group')
        .reindex(columns=labels)
    )

    if annual.empty:
        print("No annual data to plot.")
        return

    plt.figure(figsize=(10, 6))
    for label in labels:
        if label in annual.columns:
            plt.plot(annual.index, annual[label], marker='o', label=label)

    plt.xlabel('Year')
    plt.ylabel('Total water consumption per year')
    plt.title('Yearly Total Evolution of Water Consumption by Farm Size Group')
    plt.legend(title='Size Group')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    

def compare_fattening_cycles(df: pd.DataFrame, farm_id: str) -> None:

    # Safety checks
    subset = df[df['MO'] == farm_id].copy()
    if subset.empty:
        print(f"No data for farm {farm_id}")
        return
    subset['Data_censo'] = pd.to_datetime(subset['Data_censo'])
    capacities = subset['capacidad'].unique()

    # Print capacity
    print(f"\nFarm capacities found: {capacities}\n")
    cycles = subset['fecha_inicio_ciclo'].dropna().unique().tolist()
    if len(cycles) < 2:
        print("Not enough cycles to compare.")
        return
    records = []

    # Print cycles
    for c in sorted(cycles):
        sub = subset[subset['fecha_inicio_ciclo'] == c]
        end_date = sub['Data_censo'].max()
        records.append((c, end_date))
    print("Available cycles:")
    for i, (start, end) in enumerate(records, 1):
        print(f"{i}: start {start}, end {end.date()}")
    i1 = int(input("Select first cycle number: ")) - 1
    i2 = int(input("Select second cycle number: ")) - 1
    c1, e1 = records[i1]
    c2, e2 = records[i2]

    # Get required data to plot
    df1 = subset[subset['fecha_inicio_ciclo'] == c1].sort_values('Data_censo')
    df2 = subset[subset['fecha_inicio_ciclo'] == c2].sort_values('Data_censo')
    w1 = df1['Total water consumption'].values
    w2 = df2['Total water consumption'].values
    n1 = df1['num_anim_total'].values
    n2 = df2['num_anim_total'].values
    p1 = df1['total_peso'].values
    p2 = df2['total_peso'].values
    t1 = df1['t2m_C'].values
    t2 = df2['t2m_C'].values
    days = min(len(w1), len(w2), len(n1), len(n2), len(p1), len(p2), len(t1), len(t2))
    days_idx = range(1, days+1)

    # Plot Water consumption comparison
    plt.figure(figsize=(10,5))
    plt.plot(days_idx, w1[:days], label=f"Water {c1}")
    plt.plot(days_idx, w2[:days], label=f"Water {c2}")
    plt.plot(days_idx, n1[:days], '--', label=f"Animals {c1}")
    plt.plot(days_idx, n2[:days], '--', label=f"Animals {c2}")
    plt.xlabel('Day of cycle')
    plt.legend()
    plt.title(f'Water & Animals for farm {farm_id}')
    plt.tight_layout()
    plt.show()

    # Plot Water consumption comparison (bar plot)
    diff_w = w1[:days] - w2[:days]
    diff_n = n1[:days] - n2[:days]
    plt.figure(figsize=(10,5))
    plt.bar(days_idx, diff_w, alpha=0.7, label='Water diff')
    plt.bar(days_idx, diff_n, alpha=0.7, label='Animals diff')
    plt.xlabel('Day of cycle')
    plt.legend()
    plt.title('Difference per day (water vs animals)')
    plt.tight_layout()
    plt.show()

    # Plot weight comparison
    plt.figure(figsize=(10,5))
    plt.plot(days_idx, p1[:days], label=f"Weight {c1}")
    plt.plot(days_idx, p2[:days], label=f"Weight {c2}")
    plt.xlabel('Day of cycle')
    plt.ylabel('Total weight')
    plt.legend()
    plt.title('Total weight evolution comparison')
    plt.tight_layout()
    plt.show()

    # Plot temperature comparison
    plt.figure(figsize=(10,5))
    plt.plot(days_idx, t1[:days], label=f"Temp {c1}")
    plt.plot(days_idx, t2[:days], label=f"Temp {c2}")
    plt.xlabel('Day of cycle')
    plt.ylabel('t2m_C')
    plt.legend()
    plt.title('Temperature evolution comparison')
    plt.tight_layout()
    plt.show()

def summarize_consumption_by_date(df: pd.DataFrame) -> None:
    df = df.copy()
    df['Data_censo'] = pd.to_datetime(df['Data_censo'])
    user_start = input("Enter start date (YYYY-MM-DD): ")
    user_end = input("Enter end date (YYYY-MM-DD): ")
    try:
        start = pd.to_datetime(user_start)
    except:
        start = df['Data_censo'].min()
    try:
        end = pd.to_datetime(user_end)
    except:
        end = df['Data_censo'].max()
    min_date = df['Data_censo'].min()
    max_date = df['Data_censo'].max()
    if start < min_date:
        start = min_date
    if end > max_date:
        end = max_date
    if end < start:
        temp = end
        end = start
        start = temp
    print(f"\n\nWater consumption in catalonia from {start.date()} to {end.date()}\n")
    mask = (df['Data_censo'] >= start) & (df['Data_censo'] <= end)
    subset = df.loc[mask].copy()
    bins = [0, 500, 1000, 2000, 3000, float('inf')]
    labels = ['C1 <500 animals', 'C2 500-1000 animals', 'C3 1000-2000 animals',
              'C4 2000-3000 animals', 'C5 >3000 animals']
    subset['size_group'] = pd.cut(subset['num_anim_total'], bins=bins, labels=labels)
    sums = subset.groupby('size_group', observed=True)['Total water consumption'].sum()
    avgs = subset.groupby('size_group', observed=True)['Total water consumption'].mean()
    total_sum = sums.sum()
        # Print totals
    print("\n============== Total ==============\n")
    for label in labels:
        val = sums.get(label, 0)
        print(f"{label}: {val:.2e}")
    print(f"Total: {total_sum:.2e} L")
    print("\n============== Average (per day) ==============\n")
    for label in labels:
        val = avgs.get(label, 0)
        print(f"{label}: {val}")
    print(f"Total: {avgs.mean()} L per day\n\n")
