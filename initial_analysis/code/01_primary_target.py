# Loads the dataset, ensures the column AS_0S_1 exists, converts it into a binary integer called target, and raises error if the column is missing

import pandas as pd

# Load dataset (Excel file instead of CSV)
df = pd.read_excel("../data/dataset.xlsx")

# ---- Step 1a: Primary Target ----
# Make sure AS_0S_1 is binary (0 = asymptomatic, 1 = symptomatic)
if "AS_0S_1" in df.columns:
    df["target"] = df["AS_0S_1"].astype(int)
else:
    raise ValueError("AS_0S_1 column not found. Please check dataset.")

# Save back to Excel (overwrites or creates new file)
df.to_excel("../data/dataset_with_target.xlsx", index=False)