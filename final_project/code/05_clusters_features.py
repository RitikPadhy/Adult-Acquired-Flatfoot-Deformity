# all_feature_clusters.py
# Generates KDE plots showing clusters of 0s and 1s for all selected radiographic features
# and saves all of them in a single image.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Load data
df = pd.read_csv('../results/CSV/radiographic_data_preprocessed.csv')

target_col = 'target'
params = ['MCL_mean','MCL_var','1MTA_mean','1MTA_var','MT_calA_mean','MT_calA_var',
          'ML_mean','ML_var','MA_mean','MA_var','LL_mean','LL_var','5MTCA_mean','5MTCA_var',
          'cal_PA_mean','cal_PA_var','TUA_mean','TUA_var','TCA_mean','TCA_var']

# -------------------------------
# Create folder to save figure
fig_folder = '../figs'
os.makedirs(fig_folder, exist_ok=True)

# -------------------------------
# Create a grid figure (5 rows x 4 columns)
rows = 5
cols = 4
fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
axes = axes.flatten()

# Plot KDE for each feature
for i, feature in enumerate(params):
    sns.kdeplot(df[df[target_col]==0][feature], label='target=0', fill=True, ax=axes[i])
    sns.kdeplot(df[df[target_col]==1][feature], label='target=1', fill=True, ax=axes[i])
    axes[i].set_title(feature)
    axes[i].legend()

# Remove empty subplots if any
for j in range(len(params), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(os.path.join(fig_folder, 'all_feature_clusters.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ All feature clusters saved in one figure: {os.path.join(fig_folder, 'all_feature_clusters.png')}")