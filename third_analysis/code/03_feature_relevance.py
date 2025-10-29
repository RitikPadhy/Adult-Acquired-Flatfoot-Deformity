# ===============================================================
# Bland–Altman Feature Ranking
# Rank features based on agreement (closeness to red line)
# ===============================================================

import pandas as pd
import numpy as np
import os

print("🔹 Step: Computing Bland–Altman feature rankings...")

# ===============================================================
# Load dataset
# ===============================================================

data_path = '../data/dataset_with_target.xlsx'

try:
    df = pd.read_excel(data_path)
    print(f"✅ Loaded dataset: {data_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
except Exception as e:
    raise FileNotFoundError(f"❌ Could not read Excel file at {data_path}: {e}")

# ===============================================================
# Setup
# ===============================================================

all_columns = [
    "cal_PA", "MA", "TCA", "ML", "LL", "MCL",
    "MT_calA", "1MTA", "5MTCA", "TUA"
]

# Folder for output rankings
output_dir = "../figs/bland_altman_stats"
os.makedirs(output_dir, exist_ok=True)

# ===============================================================
# Helper function
# ===============================================================

def compute_agreement(df, prefix_pairs):
    """
    Given list of (label, col1, col2) tuples, compute metrics for agreement.
    Smaller SD or MAD => points closer to red line.
    """
    results = []
    for label, col1, col2 in prefix_pairs:
        if col1 not in df or col2 not in df:
            continue
        diff = df[col1] - df[col2]
        mean_diff = np.mean(diff)
        sd_diff = np.std(diff)
        mad_diff = np.mean(np.abs(diff))
        results.append({
            "Feature": label,
            "MeanDiff": mean_diff,
            "SD": sd_diff,
            "MAD": mad_diff
        })
    return pd.DataFrame(results).sort_values(by="SD", ascending=True).reset_index(drop=True)

# ===============================================================
# Prepare comparison groups
# ===============================================================

pairs_inter_T1 = [(p, f"{p}_11", f"{p}_21") for p in all_columns]   # Inter-observer (T1: KP vs KT)
pairs_inter_T2 = [(p, f"{p}_12", f"{p}_22") for p in all_columns]   # Inter-observer (T2: KP vs KT)
pairs_intra_KP = [(p, f"{p}_11", f"{p}_12") for p in all_columns]   # Intra-observer (KP: T1 vs T2)
pairs_intra_KT = [(p, f"{p}_21", f"{p}_22") for p in all_columns]   # Intra-observer (KT: T1 vs T2)

# ===============================================================
# Compute agreement metrics
# ===============================================================

df_inter_T1 = compute_agreement(df, pairs_inter_T1)
df_inter_T2 = compute_agreement(df, pairs_inter_T2)
df_intra_KP = compute_agreement(df, pairs_intra_KP)
df_intra_KT = compute_agreement(df, pairs_intra_KT)

# ===============================================================
# Save rankings to CSV
# ===============================================================

df_inter_T1.to_csv(f"{output_dir}/ranking_inter_T1.csv", index=False)
df_inter_T2.to_csv(f"{output_dir}/ranking_inter_T2.csv", index=False)
df_intra_KP.to_csv(f"{output_dir}/ranking_intra_KP.csv", index=False)
df_intra_KT.to_csv(f"{output_dir}/ranking_intra_KT.csv", index=False)

print("\n✅ Bland–Altman feature rankings saved to:")
print(f"   📁 {output_dir}")

# ===============================================================
# Print top results
# ===============================================================

print("\n=== 🔹 Top Agreement Features (Lowest SD) ===")
print("\nInter-Observer (T1):")
print(df_inter_T1.head(5).to_string(index=False))

print("\nInter-Observer (T2):")
print(df_inter_T2.head(5).to_string(index=False))

print("\nIntra-Observer (KP):")
print(df_intra_KP.head(5).to_string(index=False))

print("\nIntra-Observer (KT):")
print(df_intra_KT.head(5).to_string(index=False))

print("\n✅ Feature ranking complete!")