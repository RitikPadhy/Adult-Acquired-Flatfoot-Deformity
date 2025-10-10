# ===============================================================
# all_feature_clusters_pure_ignore_outliers.py
# Generates KDE plots for "pure" radiographic features
# OUTLIER cells are ignored
# ===============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🔹 Generating KDE plots for pure features (ignoring OUTLIERs)...")

# -------------------------------
# Load preprocessed dataset
# -------------------------------
df = pd.read_csv('../results/CSV/radiographic_data_preprocessed.csv')
target_col = 'target'

# -------------------------------
# Select only "pure" columns
# -------------------------------
pure_features = [col for col in df.columns if 'pure' in col.lower()]
if len(pure_features) == 0:
    raise ValueError("No columns with 'pure' found in dataset.")

print(f"📊 Pure features selected for KDE plots: {len(pure_features)}")

# -------------------------------
# Create folder to save figure
# -------------------------------
fig_folder = '../figs/feature_clusters'
os.makedirs(fig_folder, exist_ok=True)

# -------------------------------
# Create a grid figure dynamically
# -------------------------------
num_features = len(pure_features)
cols = 4
rows = (num_features + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*3))
axes = axes.flatten()

# -------------------------------
# Plot KDE for each pure feature, ignoring OUTLIER cells
# -------------------------------
for i, feature in enumerate(pure_features):
    # Convert numeric, non-numeric (OUTLIER) will become NaN
    numeric_feature = pd.to_numeric(df[feature], errors='coerce')
    
    # Plot KDE for target=0
    valid_0 = numeric_feature[df[target_col]==0].dropna()
    if len(valid_0) > 0:
        sns.kdeplot(valid_0, label='target=0', fill=True, ax=axes[i])
    
    # Plot KDE for target=1
    valid_1 = numeric_feature[df[target_col]==1].dropna()
    if len(valid_1) > 0:
        sns.kdeplot(valid_1, label='target=1', fill=True, ax=axes[i])
    
    axes[i].set_title(feature)
    axes[i].legend()

# Remove any empty subplots
for j in range(len(pure_features), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()

# -------------------------------
# Save figure
# -------------------------------
fig_path = os.path.join(fig_folder, 'all_pure_feature_clusters_ignore_outliers.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ All pure feature clusters saved in one figure: {fig_path}")