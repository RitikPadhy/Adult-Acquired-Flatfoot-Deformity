# Generates a combined feature importance chart (Correlation, Mutual Info, Random Forest Importance)
# for selected radiographic features and saves the plot in ../figs/feature_importance

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier

# Load data
df = pd.read_csv('../results/CSV/radiographic_data_preprocessed.csv')

# Define target and selected radiographic features
target_col = 'target'
params = ['MCL_mean','MCL_var','1MTA_mean','1MTA_var','MT_calA_mean','MT_calA_var',
          'ML_mean','ML_var','MA_mean','MA_var','LL_mean','LL_var','5MTCA_mean','5MTCA_var',
          'cal_PA_mean','cal_PA_var','TUA_mean','TUA_var','TCA_mean','TCA_var']

X = df[params]
y = df[target_col]

# Create folder for figures
fig_folder = '../figs/feature_importance'
os.makedirs(fig_folder, exist_ok=True)

# 1️⃣ Compute Point-Biserial Correlation
correlations = {}
for col in X.columns:
    corr, _ = pointbiserialr(y, X[col])
    correlations[col] = corr
corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])

# 2️⃣ Compute Mutual Information
mi_scores = mutual_info_classif(X, y, random_state=0)
mi_df = pd.DataFrame({'Mutual_Info': mi_scores}, index=params)

# 3️⃣ Compute Random Forest Feature Importance
rf_model = RandomForestClassifier(random_state=0)
rf_model.fit(X, y)
rf_importances = pd.Series(rf_model.feature_importances_, index=params, name='RF_Importance')

# Combine all metrics into a single DataFrame
importance_df = pd.concat([corr_df, mi_df, rf_importances], axis=1)
importance_df = importance_df.sort_values(by='Mutual_Info', ascending=False)  # optional: sort by strongest feature

# 4️⃣ Plot combined feature importance
fig, ax = plt.subplots(figsize=(12,6))
importance_df.plot(kind='bar', ax=ax)
ax.set_title('Feature Importance: Correlation, Mutual Info, RF Importance')
ax.set_ylabel('Score')
ax.set_xlabel('Feature')
plt.xticks(rotation=45, ha='right')

# Save figure
fig.savefig(os.path.join(fig_folder, 'combined_feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"✅ Combined feature importance figure saved in {fig_folder}")