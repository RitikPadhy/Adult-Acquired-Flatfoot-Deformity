# ===============================================================
# Radiographic Data Aggregation (Four Reliability Groups)
# ===============================================================

import pandas as pd
import numpy as np
import os

print("🔹Aggregating repeated measurements by reliability pairs...")

# Load dataset
df = pd.read_excel("../data/dataset_with_target.xlsx")

# Identify base parameters
param_cols = df.columns
base_params = set([col.rsplit("_", 1)[0] for col in param_cols if col.endswith(("11", "12", "21", "22"))])

# Define comparison pairs
pairs = {
    "first_time": ["11", "21"],
    "second_time": ["12", "22"],
    "observer1": ["11", "12"],
    "observer2": ["21", "22"]
}

# Compute mean and variance for each pair
for param in base_params:
    for label, (a, b) in pairs.items():
        cols = [f"{param}_{a}", f"{param}_{b}"]
        available = [c for c in cols if c in df.columns]
        
        if len(available) == 2:
            df[f"{param}_{label}_mean"] = df[available].mean(axis=1)
            df[f"{param}_{label}_var"] = df[available].var(axis=1)

# Save processed dataset
os.makedirs("../results/Excel", exist_ok=True)
os.makedirs("../results/CSV", exist_ok=True)

df.to_excel("../results/Excel/radiographic_data_aggregated_pairs.xlsx", index=False)
df.to_csv("../results/CSV/radiographic_data_aggregated_pairs.csv", index=False)

print("✅ Pairwise aggregation done. Files saved to /results/Excel and /results/CSV.")