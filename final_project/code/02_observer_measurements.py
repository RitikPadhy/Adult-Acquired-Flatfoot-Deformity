# Computes the mean and variance for the groups of radiographic parameters with suffixes _11, _12, _21, _22 , and then saves the result in both csv and excel file
# Results say that the aggregation completed successfully

import pandas as pd
import numpy as np

# Load dataset
df = pd.read_excel("../data/dataset_with_target.xlsx")

# Identify unique parameter base names (strip _11, _12, _21, _22)
param_cols = df.columns
base_params = set([col.rsplit("_", 1)[0] for col in param_cols if col.endswith(("11","12","21","22"))])

# Aggregate mean and variance
for param in base_params:
    cols = [f"{param}_11", f"{param}_12", f"{param}_21", f"{param}_22"]
    available = [c for c in cols if c in df.columns]  # in case of missing columns
    
    if len(available) > 0:
        df[f"{param}_mean"] = df[available].mean(axis=1)
        df[f"{param}_var"] = df[available].var(axis=1)

# Save processed dataset to Excel
df.to_excel("../results/Excel/radiographic_data_aggregated.xlsx", index=False)

print("✅ Aggregation done. New file: radiographic_data_aggregated.xlsx")
print("New columns added: *_mean, *_var")

# Save processed dataset to CSV
df.to_csv("../results/CSV/radiographic_data_aggregated.csv", index=False)

print("✅ Aggregation done. New file: radiographic_data_aggregated.csv")
print("New columns added: *_mean, *_var")