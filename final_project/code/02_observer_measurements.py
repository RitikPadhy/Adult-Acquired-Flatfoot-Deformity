# ===============================================================
# Radiographic Data Aggregation + ICC + Bland-Altman Analysis
# ===============================================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ===============================================================
# STEP 1: Aggregation (Compute mean and variance)
# ===============================================================
print("🔹 Step 1: Aggregating repeated measurements...")

# Load dataset
df = pd.read_excel("../data/dataset_with_target.xlsx")

# Identify base parameters
param_cols = df.columns
base_params = set([col.rsplit("_", 1)[0] for col in param_cols if col.endswith(("11","12","21","22"))])

# Compute mean and variance
for param in base_params:
    cols = [f"{param}_11", f"{param}_12", f"{param}_21", f"{param}_22"]
    available = [c for c in cols if c in df.columns]  # only use existing columns
    
    if len(available) > 0:
        df[f"{param}_mean"] = df[available].mean(axis=1)
        df[f"{param}_var"] = df[available].var(axis=1)

# Save processed dataset
os.makedirs("../results/Excel", exist_ok=True)
os.makedirs("../results/CSV", exist_ok=True)

df.to_excel("../results/Excel/radiographic_data_aggregated.xlsx", index=False)
df.to_csv("../results/CSV/radiographic_data_aggregated.csv", index=False)

print("✅ Aggregation done. Files saved to /results/Excel and /results/CSV.")

# ===============================================================
# STEP 2: ICC Computation
# ===============================================================
print("\n🔹 Step 2: Computing ICC values...")

# Reload from cleaned dataset (or use aggregated df)
df = pd.read_csv('../results/CSV/radiographic_data_aggregated.csv')

# Define all 10 angles and their repeated measurement columns
angle_columns = {
    "cal_PA":   ['cal_PA_11','cal_PA_12','cal_PA_21','cal_PA_22'],
    "MA":       ['MA_11','MA_12','MA_21','MA_22'],
    "TUA":      ['TUA_11','TUA_12','TUA_21','TUA_22'],
    "1MTA":     ['1MTA_11','1MTA_12','1MTA_21','1MTA_22'],
    "TCA":      ['TCA_11','TCA_12','TCA_21','TCA_22'],
    "ML":       ['ML_11','ML_12','ML_21','ML_22'],
    "LL":       ['LL_11','LL_12','LL_21','LL_22'],
    "MCL":      ['MCL_11','MCL_12','MCL_21','MCL_22'],
    "MT_calA":  ['MT_calA_11','MT_calA_12','MT_calA_21','MT_calA_22'],
    "5MTCA":    ['5MTCA_11','5MTCA_12','5MTCA_21','5MTCA_22']
}

icc_results = []

for angle, cols in angle_columns.items():
    # Skip missing angles
    if not all(c in df.columns for c in cols):
        print(f"Skipping {angle}: Missing columns.")
        continue

    long_df = pd.melt(
        df,
        id_vars=['HN'],
        value_vars=cols,
        var_name='measurement',
        value_name='value'
    )
    
    long_df['rater'] = long_df['measurement'].apply(lambda x: 'KP' if '1' in x else 'KT')
    long_df['timepoint'] = long_df['measurement'].apply(lambda x: 'T1' if '1' in x.split('_')[1] else 'T2')
    long_df = long_df.dropna(subset=['value'])
    
    icc = pg.intraclass_corr(
        data=long_df,
        targets='HN',
        raters='measurement',
        ratings='value',
        nan_policy='omit'
    )
    icc['angle'] = angle
    icc_results.append(icc)

icc_all = pd.concat(icc_results, ignore_index=True)

# Save ICC results
out_file = "../results/icc_results.csv"
with open(out_file, "w") as f:
    f.write("Type,Description,ICC,F,df1,df2,pval,CI95%,angle\n")
    for angle, group in icc_all.groupby("angle"):
        group.to_csv(f, header=False, index=False, lineterminator="\n")
        f.write("\n")

print(f"✅ ICC results saved to {out_file}")

# ===============================================================
# STEP 3: Bland-Altman Analysis
# ===============================================================
print("\n🔹 Step 3: Generating Bland-Altman plots...")

def bland_altman_plot(measure1, measure2, label, plot_type="inter"):
    mean = (measure1 + measure2) / 2
    diff = measure1 - measure2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    plt.scatter(mean, diff)
    plt.axhline(md, color='red')
    plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
    plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
    plt.title(f'Bland-Altman: {label}')

    # choose folder
    folder = "../figs/intra-observer-variation" if plot_type == "intra" else "../figs/inter-observer-variation"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f'bland_altman_{label}.png'))
    plt.close()

all_columns = [
    "cal_PA", "MA", "TCA", "ML", "LL", "MCL", 
    "MT_calA", "1MTA", "5MTCA", "TUA"
]

for prefix in all_columns:
    try:
        bland_altman_plot(df[f"{prefix}_11"], df[f"{prefix}_21"], f"{prefix}_T1_KP_vs_KT", plot_type="inter")
        bland_altman_plot(df[f"{prefix}_12"], df[f"{prefix}_22"], f"{prefix}_T2_KP_vs_KT", plot_type="inter")
        bland_altman_plot(df[f"{prefix}_11"], df[f"{prefix}_12"], f"{prefix}_KP_T1_vs_T2", plot_type="intra")
        bland_altman_plot(df[f"{prefix}_21"], df[f"{prefix}_22"], f"{prefix}_KT_T1_vs_T2", plot_type="intra")
    except KeyError as e:
        print(f"Skipping {prefix}: missing column -> {e}")

print("✅ Bland-Altman plots generated and saved successfully.")