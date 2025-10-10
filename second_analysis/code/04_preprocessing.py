# ===============================================================
# Radiographic Data Preprocessing
# Handles: normalization of pure numeric values only
# OUTLIER cells are never touched, and all other columns remain intact
# ===============================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

print("🔹 Starting data preprocessing (pure columns only)...")

# ---------------------------------------------------------------
# Load cleaned dataset (after Bland–Altman filtering)
# ---------------------------------------------------------------
df = pd.read_csv('../results/CSV/radiographic_data_cleaned.csv')

# ---------------------------------------------------------------
# Identify pure columns
# ---------------------------------------------------------------
pure_cols = [col for col in df.columns if 'pure' in col.lower()]
print(f"📊 Pure columns detected: {len(pure_cols)} -> {pure_cols}")

# ---------------------------------------------------------------
# Normalize pure columns (ignore OUTLIER cells)
# ---------------------------------------------------------------
for col in pure_cols:
    # Create a mask for numeric values only (exclude OUTLIER)
    numeric_mask = pd.to_numeric(df[col], errors='coerce').notna()
    
    # Only normalize numeric entries
    if numeric_mask.sum() > 0:
        scaler = StandardScaler()
        df.loc[numeric_mask, col] = scaler.fit_transform(df.loc[numeric_mask, [col]])

# ---------------------------------------------------------------
# Save preprocessed dataset
# ---------------------------------------------------------------
os.makedirs('../results/CSV', exist_ok=True)
os.makedirs('../results/Excel', exist_ok=True)

csv_path = '../results/CSV/radiographic_data_preprocessed.csv'
excel_path = '../results/Excel/radiographic_data_preprocessed.xlsx'

df.to_csv(csv_path, index=False)
df.to_excel(excel_path, index=False)

# ---------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------
print("✅ Preprocessing completed successfully.")
print(f"📁 Files saved:\n   - {csv_path}\n   - {excel_path}")
print(f"ℹ️ Pure columns normalized (OUTLIER cells untouched): {pure_cols}")