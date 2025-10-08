# ===============================================================
# Radiographic Data Aggregation
# ===============================================================

import pandas as pd
import numpy as np
import os

print("🔹Aggregating repeated measurements...")

# Load dataset
df = pd.read_excel("../data/dataset_with_target.xlsx")

# Identify base parameters
param_cols = df.columns
base_params = set([col.rsplit("_", 1)[0] for col in param_cols if col.endswith(("11","12","21","22"))])

# Compute mean and variance
for param in base_params:
    cols = [f"{param}_11", f"{param}_12", f"{param}_21", f"{param}_22"]
    available = [c for c in cols if c in df.columns]  # only use existing columns
    
    if len(available) > 0:
        df[f"{param}_mean"] = df[available].mean(axis=1)
        df[f"{param}_var"] = df[available].var(axis=1)

# Save processed dataset
os.makedirs("../results/Excel", exist_ok=True)
os.makedirs("../results/CSV", exist_ok=True)

df.to_excel("../results/Excel/radiographic_data_aggregated.xlsx", index=False)
df.to_csv("../results/CSV/radiographic_data_aggregated.csv", index=False)

print("✅ Aggregation done. Files saved to /results/Excel and /results/CSV.")