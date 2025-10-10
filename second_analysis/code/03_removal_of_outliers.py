# ===============================================================
# Radiographic Data Cleaning & Pure Column Generation (Bland–Altman Method)
# ===============================================================

import pandas as pd
import numpy as np
import os

print("🔹 Starting Bland–Altman based outlier filtering...")

# ---------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------
df = pd.read_excel("../data/dataset_with_target.xlsx")

# ---------------------------------------------------------------
# Helper function to apply Bland–Altman filtering
# ---------------------------------------------------------------
def bland_altman_clean(df, col1, col2):
    """
    Removes outliers based on Bland–Altman limits of agreement.
    Returns: mean_diff, SD, upper, lower, pure_col (Series)
    """
    # Compute mean and difference
    mean_vals = (df[col1] + df[col2]) / 2
    diff_vals = df[col1] - df[col2]

    # Compute Bland–Altman stats
    mean_diff = diff_vals.mean()
    sd_diff = diff_vals.std()
    upper = mean_diff + 1.96 * sd_diff
    lower = mean_diff - 1.96 * sd_diff

    # Identify outliers
    mask_outlier = (diff_vals < lower) | (diff_vals > upper)

    # Create a "pure" column
    pure_col = mean_vals.copy()
    pure_col[mask_outlier] = "OUTLIER"  # Mark outliers

    return mean_diff, sd_diff, upper, lower, pure_col

# ---------------------------------------------------------------
# Identify base parameters
# ---------------------------------------------------------------
param_cols = df.columns
base_params = set([col.rsplit("_", 1)[0] for col in param_cols if col.endswith(("11","12","21","22"))])

# ---------------------------------------------------------------
# Process each parameter (angle)
# ---------------------------------------------------------------
for param in base_params:
    print(f"⚙️ Processing parameter: {param}")

    # Retrieve available columns for this parameter
    c11, c12, c21, c22 = f"{param}_11", f"{param}_12", f"{param}_21", f"{param}_22"
    available = [c for c in [c11, c12, c21, c22] if c in df.columns]

    if len(available) < 2:
        print(f"   ⚠️ Skipping {param} — not enough columns for comparison.")
        continue

    # --- Inter-observer reliability ---
    if c11 in df.columns and c21 in df.columns:
        _, _, _, _, df[f"{param}_pure_inter_first"] = bland_altman_clean(df, c11, c21)
    if c12 in df.columns and c22 in df.columns:
        _, _, _, _, df[f"{param}_pure_inter_second"] = bland_altman_clean(df, c12, c22)

    # --- Intra-observer reliability ---
    if c11 in df.columns and c12 in df.columns:
        _, _, _, _, df[f"{param}_pure_intra_obs1"] = bland_altman_clean(df, c11, c12)
    if c21 in df.columns and c22 in df.columns:
        _, _, _, _, df[f"{param}_pure_intra_obs2"] = bland_altman_clean(df, c21, c22)

# ---------------------------------------------------------------
# Save processed dataset
# ---------------------------------------------------------------
os.makedirs("../results/Excel", exist_ok=True)
os.makedirs("../results/CSV", exist_ok=True)

excel_path = "../results/Excel/radiographic_data_cleaned.xlsx"
csv_path = "../results/CSV/radiographic_data_cleaned.csv"

df.to_excel(excel_path, index=False)
df.to_csv(csv_path, index=False)

print("✅ Outlier filtering complete.")
print(f"📁 Files saved:\n   - {excel_path}\n   - {csv_path}")
