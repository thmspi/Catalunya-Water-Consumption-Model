import pandas as pd
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

original_data = os.path.join(script_dir, '..', '..', 'data','temp_results', '0230GI.csv')
original_data = os.path.abspath(original_data)

synthetic_data = os.path.join(script_dir, '..', '..', 'data','base_data', 'synthetic_data.csv')
synthetic_data = os.path.abspath(synthetic_data)

results_data = os.path.join(script_dir, '..', '..', 'data','base_data', 'merged_cleaned.csv')
results_data = os.path.abspath(results_data)


# --- load ---
orig = pd.read_csv(original_data)
synth = pd.read_csv(synthetic_data)

# Drop accidental unnamed index column in original, if present
orig = orig.loc[:, ~orig.columns.str.contains(r"^Unnamed")]

# --- normalize dates ---
orig["Date"] = pd.to_datetime(orig["Date"])
synth["Date"] = pd.to_datetime(synth["Date"])

# --- fix decimal comma in synthetic avg_weight (e.g., "137,1") ---
if "avg_weight" in synth.columns:
    synth["avg_weight"] = (
        synth["avg_weight"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )

# --- keep only common columns ---
common_cols = sorted(set(orig.columns) & set(synth.columns))
# (From your samples, likely: ['Date','water_L','avg_weight','total_weight','num_anim_total','t2m_C'])

orig_c = orig[common_cols].copy()
synth_c = synth[common_cols].copy()

# Ensure numeric types where appropriate (optional but helpful)
for col in ["water_L", "avg_weight", "total_weight", "num_anim_total", "t2m_C"]:
    if col in orig_c.columns:
        orig_c[col] = pd.to_numeric(orig_c[col], errors="coerce")
    if col in synth_c.columns:
        synth_c[col] = pd.to_numeric(synth_c[col], errors="coerce")

# --- upsert by Date ---
orig_c = orig_c.set_index("Date").sort_index()
synth_c = synth_c.set_index("Date").sort_index()

# 1) Update existing dates in orig with values from synth (only in common cols)
orig_c.update(synth_c)

# 2) Append rows that exist only in synthetic
new_dates = synth_c.index.difference(orig_c.index)
result = pd.concat([orig_c, synth_c.loc[new_dates]], axis=0).sort_index()

# --- save ---
result.reset_index().to_csv(results_data, index=False)
print(f"Saved {len(result)} rows with common columns to: {results_data}")
