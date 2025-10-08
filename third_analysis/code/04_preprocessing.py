# ===============================================================
# Radiographic Data Preprocessing
# Handles missing values, standardizes only mean/variance features,
# encodes categorical variables, and saves the cleaned dataset.
# ===============================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

print("🔹Starting preprocessing...")

# Load dataset
df = pd.read_csv('../results/CSV/radiographic_data_aggregated_pairs.csv')

# ---------------------------
# Handle missing values
print("Handling missing values...")
df = df[df.isnull().mean(axis=1) < 0.3]  # Drop rows with >30% missing
df = df.fillna(df.median(numeric_only=True))  # Impute remaining with median

# ---------------------------
# Select only mean/variance columns for normalization
mean_var_cols = [col for col in df.columns if col.endswith(('_mean', '_var'))]
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Detected {len(mean_var_cols)} mean/variance numeric columns for scaling.")
print(f"Detected {len(categorical_cols)} categorical columns for encoding.")

# ---------------------------
# Standardize mean/variance features
if mean_var_cols:
    print("Standardizing mean/variance features...")
    scaler = StandardScaler()
    df[mean_var_cols] = scaler.fit_transform(df[mean_var_cols])

# ---------------------------
# Encode categorical features
if categorical_cols:
    print("Encoding categorical features...")
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))

# ---------------------------
# Save cleaned dataset
df.to_csv('../results/CSV/radiographic_data_preprocessed.csv', index=False)
df.to_excel('../results/Excel/radiographic_data_preprocessed.xlsx', index=False)

print("✅ Preprocessing completed. Only mean/variance columns normalized. Files saved to /results/CSV and /results/Excel.")