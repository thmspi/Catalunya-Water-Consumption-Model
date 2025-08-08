import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', '..', 'data','models', 'knn_water_model.joblib')
csv_path = os.path.abspath(csv_path)
model = joblib.load(csv_path)


def formatting_dataset(data):

    formatted_data = data.copy()

    columns = [
        "row",
        "MO",
        "Insertion date",
        "capacity",
        "in",
        "out",
        "in_animals",
        "out_animals",
        "num_anim_total",
        "fatting-start",
        "month-year-fatting-start",
        "week",
        "avg_weight",
        "total_weight",
        "max_mov",
        "deaths",
        "death-percentage",
        "pes_acum_morts",
        "estimations",
        "farm_status",
        "cycle_id",
        "cycle-change",
        "origins",
        "farm_origins",
        "full_origins",
        "destination",
        "full_destination",
        "nb_anim_final",
        "acum_estimats",
        "exces_manca",
        "t2m_C"
    ]
    formatted_data.columns = columns
    formatted_data.head()

    print(formatted_data.dtypes)

    formatted_data['Insertion date'] = pd.to_datetime(formatted_data['Insertion date'], errors='coerce')

    return formatted_data



def create_subset(data: pd.DataFrame,
                  results: pd.DataFrame,
                  farm_id: str):
    # filter and create subset with only required columns
    subset = data.loc[data["MO"] == farm_id].copy()
    subset = subset[["Insertion date", "total_weight", "num_anim_total", "t2m_C"]]

    if subset.empty:
        print(f"No rows found for farm ID {farm_id!r}")
        return None, results

    # Set date (month) to the required format
    subset['month']     = subset['Insertion date'].dt.month
    subset['month_sin'] = np.sin(2 * np.pi * subset['month'] / 12)
    subset['month_cos'] = np.cos(2 * np.pi * subset['month'] / 12)

    temp_result = pd.DataFrame({
        "Insertion date": subset["Insertion date"].values,
        "MO": farm_id
    })

    # Concat to create our subset
    results = pd.concat([results, temp_result], ignore_index=True)

    # Drop non-required variables
    subset = subset.drop(columns=["Insertion date", "month"])

    return subset, results


def adapt_subset_to_model(subset: pd.DataFrame):
    to_scale = [
        'num_anim_total',
        'total_weight',
        "t2m_C"
    ]   
    
    pipe = Pipeline([
        ('std', StandardScaler()),
        ('norm', MinMaxScaler())
    ])

    subset[to_scale] = pipe.fit_transform(subset[to_scale])

    return subset

def predict_farm_water_consumption(subset: pd.DataFrame,  results: pd.DataFrame):
    feature_cols = [
        "total_weight",
        "num_anim_total",
        "t2m_C",
        "month_sin",
        "month_cos"
    ]

    results["Total water consumption"] = model.predict(subset[feature_cols])

    return results

def farm_prediction( data: pd.DataFrame, farm_id: str):
    
    results = pd.DataFrame(columns=['MO','Insertion date', 'Total water consumption'])
    
    # 1) Create a subset for the specific farm
    subset, results = create_subset(data, results, farm_id)

    # 2) Transform variables into the model learning format
    subset = adapt_subset_to_model(subset)
    
    # 3) Predict water consumption and add to the original data
    results = predict_farm_water_consumption(subset, results)


    return results