# In[1]: Importing necessary libraries
import pandas as pd
import warnings
import os

warnings.filterwarnings("ignore")

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, '..', '..', 'data','temp_results', 'census_all_t2m.csv')
csv_path = os.path.abspath(csv_path)

# In[2]: Reading and filtering the dataset
data = pd.read_csv(csv_path)

farm_code = "0230GI" # Maria's farm
data = data.loc[data["MO"] == farm_code].copy()

data = data.rename(columns={"Data_censo": "Insertion date"})

# In[3]: Rename columns to English for clarity
english_cols = [
    "row_id",
    "MO",
    "Insertion date",
    "capacity",
    "id_mov_entrada",
    "id_mov_salida",
    "num_anim_entrada",
    "num_anim_salida",
    "num_anim_total",
    "fecha_inicio_ciclo",
    "periodo_inicio_ciclo",
    "semana",
    "peso_medio",
    "total_peso",
    "max_mov",
    "morts",
    "percent_mortalitat",
    "pes_acum_morts",
    "estimats",
    "estat_explotacio",
    "cicle",
    "canvi_cicle",
    "origens",
    "explot_origens",
    "acum_origens",
    "explot_desti",
    "acum_destins",
    "num_anim_final",
    "acum_estimats",
    "exces_manca",
    "t2m_C"
]
data.columns = english_cols

# In[4]: Parse dates
data["Insertion date"] = pd.to_datetime(data["Insertion date"], errors="coerce")

# In[5]: Subset to the desired date range
subset = data[
    (data["Insertion date"] >= "2021-04-26") &
    (data["Insertion date"] <= "2025-07-01")
].copy()

# In[6]: Keep only the variables you care about
subset = subset[[
    "Insertion date",
    "peso_medio",       # avg_weight
    "total_peso",       # total_weight
    "num_anim_total",
    "t2m_C"
]]

# rename for clarity
subset = subset.rename(columns={
    "Insertion date": "Date",
    "peso_medio":     "avg_weight",
    "total_peso":     "total_weight"
})

# In[7]: Export

out_path = os.path.join(script_dir, '..', '..', 'data','temp_results','filtered_census_t2m.csv')
out_path = os.path.abspath(out_path)

subset.to_csv(out_path, index=False)
print(f"Written {len(subset)} rows for farm {farm_code} to {out_path}")
