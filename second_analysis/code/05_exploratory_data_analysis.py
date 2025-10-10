# ===============================================================
# Combined Feature Importance for Pure Radiographic Features
# Ignores OUTLIER cells in pure columns
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

print("🔹 Generating feature importance for 'pure' columns (ignoring OUTLIERs)...")

# ---------------------------------------------------------------
# Load preprocessed dataset
# ---------------------------------------------------------------
df = pd.read_csv('../results/CSV/radiographic_data_preprocessed.csv')

# ---------------------------------------------------------------
# Select target and pure features
# ---------------------------------------------------------------
target_col = 'target'
pure_features = [col for col in df.columns if 'pure' in col.lower()]
if len(pure_features) == 0:
    raise ValueError("No columns with 'pure' found in dataset.")

# ---------------------------------------------------------------
# Create folder for figures
# ---------------------------------------------------------------
fig_folder = '../figs/feature_importance'
os.makedirs(fig_folder, exist_ok=True)

# ---------------------------------------------------------------
# 1️⃣ Compute Point-Biserial Correlation
# ---------------------------------------------------------------
correlations = {}
for col in pure_features:
    # Only use numeric values, ignore OUTLIERs
    numeric_values = pd.to_numeric(df[col], errors='coerce')
    valid_mask = numeric_values.notna()
    if valid_mask.sum() > 1:
        corr, _ = pointbiserialr(df.loc[valid_mask, target_col], numeric_values[valid_mask])
        correlations[col] = corr
    else:
        correlations[col] = float('nan')

corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])

# ---------------------------------------------------------------
# 2️⃣ Compute Mutual Information
# ---------------------------------------------------------------
mi_scores = []
for col in pure_features:
    numeric_values = pd.to_numeric(df[col], errors='coerce')
    valid_mask = numeric_values.notna()
    if valid_mask.sum() > 1:
        mi_score = mutual_info_classif(
            numeric_values[valid_mask].values.reshape(-1,1),
            df.loc[valid_mask, target_col],
            random_state=0
        )[0]
    else:
        mi_score = float('nan')
    mi_scores.append(mi_score)

mi_df = pd.DataFrame({'Mutual_Info': mi_scores}, index=pure_features)

# ---------------------------------------------------------------
# 3️⃣ Compute Random Forest Feature Importance
# ---------------------------------------------------------------
rf_importances = []
for col in pure_features:
    numeric_values = pd.to_numeric(df[col], errors='coerce')
    valid_mask = numeric_values.notna()
    if valid_mask.sum() > 1:
        rf_model = RandomForestClassifier(random_state=0)
        rf_model.fit(numeric_values[valid_mask].values.reshape(-1,1),
                     df.loc[valid_mask, target_col])
        rf_importances.append(rf_model.feature_importances_[0])
    else:
        rf_importances.append(float('nan'))

rf_df = pd.DataFrame({'RF_Importance': rf_importances}, index=pure_features)

# ---------------------------------------------------------------
# Combine all metrics into a single DataFrame
# ---------------------------------------------------------------
importance_df = pd.concat([corr_df, mi_df, rf_df], axis=1)

# ---------------------------------------------------------------
# 4️⃣ Plot combined feature importance
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12,6))
importance_df.plot(kind='bar', ax=ax)
ax.set_title('Feature Importance: Correlation, Mutual Info, RF Importance (Pure Features, OUTLIERs ignored)')
ax.set_ylabel('Score')
ax.set_xlabel('Feature')
plt.xticks(rotation=45, ha='right')

# Save figure
fig_path = os.path.join(fig_folder, 'combined_feature_importance_pure.png')
fig.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ Combined feature importance figure saved: {fig_path}")