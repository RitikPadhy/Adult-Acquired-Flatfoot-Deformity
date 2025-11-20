# ===============================================================
# Bland–Altman Feature Ranking (Using MeanDiff, SD, MAD)
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

output_dir = "../figs/bland_altman_stats"
os.makedirs(output_dir, exist_ok=True)

# ===============================================================
# Helper: normalize a column
# ===============================================================

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

# ===============================================================
# Helper function to compute agreement and ranking
# ===============================================================

def compute_agreement(df, prefix_pairs):
    """
    Compute MeanDiff, SD, MAD and combine all three for ranking.
    Lower CombinedScore => better agreement.
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

    df_res = pd.DataFrame(results)

    # Normalize all metrics
    df_res["Norm_SD"] = normalize(df_res["SD"])
    df_res["Norm_MAD"] = normalize(df_res["MAD"])
    df_res["Norm_MeanDiff"] = normalize(abs(df_res["MeanDiff"]))

    # Combined ranking score
    df_res["CombinedScore"] = (
        df_res["Norm_SD"] +
        df_res["Norm_MAD"] +
        df_res["Norm_MeanDiff"]
    )

    # Sort by combined metric
    return df_res.sort_values(by="CombinedScore", ascending=True).reset_index(drop=True)

# ===============================================================
# Prepare comparison groups
# ===============================================================

pairs_inter_T1 = [(p, f"{p}_11", f"{p}_21") for p in all_columns]
pairs_inter_T2 = [(p, f"{p}_12", f"{p}_22") for p in all_columns]
pairs_intra_KP = [(p, f"{p}_11", f"{p}_12") for p in all_columns]
pairs_intra_KT = [(p, f"{p}_21", f"{p}_22") for p in all_columns]

# ===============================================================
# Compute agreement metrics + combined score ranking
# ===============================================================

df_inter_T1 = compute_agreement(df, pairs_inter_T1)
df_inter_T2 = compute_agreement(df, pairs_inter_T2)
df_intra_KP = compute_agreement(df, pairs_intra_KP)
df_intra_KT = compute_agreement(df, pairs_intra_KT)

# ===============================================================
# Save results
# ===============================================================

df_inter_T1.to_csv(f"{output_dir}/ranking_inter_T1.csv", index=False)
df_inter_T2.to_csv(f"{output_dir}/ranking_inter_T2.csv", index=False)
df_intra_KP.to_csv(f"{output_dir}/ranking_intra_KP.csv", index=False)
df_intra_KT.to_csv(f"{output_dir}/ranking_intra_KT.csv", index=False)

print("\n✅ Bland–Altman feature rankings saved to:", output_dir)

# ===============================================================
# Print top results
# ===============================================================

print("\n=== 🔹 Top Features (Lowest CombinedScore) ===")
print("\nInter-Observer (T1):")
print(df_inter_T1.head(5).to_string(index=False))

print("\nInter-Observer (T2):")
print(df_inter_T2.head(5).to_string(index=False))

print("\nIntra-Observer (KP):")
print(df_intra_KP.head(5).to_string(index=False))

print("\nIntra-Observer (KT):")
print(df_intra_KT.head(5).to_string(index=False))

print("\n✅ Feature ranking complete!")