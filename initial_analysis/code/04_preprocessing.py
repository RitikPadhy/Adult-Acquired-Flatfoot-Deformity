# Handles missing values, standardizing/normalizing continuous radiographic features, encoding categorical variables, and saving the cleaned dataset for later use.
# Results suggest preprocessing ran successfully


import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataframe
df = pd.read_csv('../results/CSV/radiographic_data_aggregated.csv')

# ---------------------------
# Handle missing values
df = df[df.isnull().mean(axis=1) < 0.3]  # Drop rows with >30% missing
df = df.fillna(df.median(numeric_only=True))  # Impute remaining with median

# ---------------------------
# Standardize continuous features
continuous_features = [
    'MCL_mean', 'MCL_var', '1MTA_mean', '1MTA_var', 
    'MT_calA_mean', 'MT_calA_var', 'ML_mean', 'ML_var', 
    'MA_mean', 'MA_var', 'LL_mean', 'LL_var', 
    '5MTCA_mean', '5MTCA_var', 'cal_PA_mean', 'cal_PA_var', 
    'TUA_mean', 'TUA_var', 'TCA_mean', 'TCA_var'
]

scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# ---------------------------
# Encode categorical features
categorical_features = ['M_1 F_2', 'CMP_1A_2']  # Adjust columns as needed
le = LabelEncoder()
for col in categorical_features:
    if col in df.columns:
        df[col] = le.fit_transform(df[col])

# Save the preprocessed dataframe for later use
df.to_csv('../results/CSV/radiographic_data_preprocessed.csv', index=False)
print("Preprocessing completed and saved to radiographic_data_preprocessed.csv")

# Save the preprocessed dataframe for later use
df.to_excel('../results/Excel/radiographic_data_preprocessed.xlsx', index=False)
print("Preprocessing completed and saved to radiographic_data_preprocessed.excel")