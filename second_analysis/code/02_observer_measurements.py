# ===============================================================
# Bland–Altman Analysis (Inter- and Intra-Observer Variation)
# Combined plots: one figure for inter, one for intra
# ===============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

print("🔹 Step: Generating combined Bland–Altman plots...")

# Load dataset
df = pd.read_csv('../results/CSV/radiographic_data_aggregated.csv')

# ===============================================================
# Bland–Altman plotting helper
# ===============================================================

def bland_altman_plot(ax, measure1, measure2, title):
    """Draw a single Bland–Altman plot on a provided axis."""
    mean = (measure1 + measure2) / 2
    diff = measure1 - measure2
    md = np.mean(diff)
    sd = np.std(diff)

    ax.scatter(mean, diff, alpha=0.6, s=15)
    ax.axhline(md, color='red', linewidth=1)
    ax.axhline(md + 1.96*sd, color='blue', linestyle='--', linewidth=0.8)
    ax.axhline(md - 1.96*sd, color='blue', linestyle='--', linewidth=0.8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(axis='both', labelsize=7)
    ax.grid(alpha=0.3)

# ===============================================================
# Setup
# ===============================================================

all_columns = [
    "cal_PA", "MA", "TCA", "ML", "LL", "MCL", 
    "MT_calA", "1MTA", "5MTCA", "TUA"
]

# Create folders
os.makedirs("../figs/inter-observer-variation", exist_ok=True)
os.makedirs("../figs/intra-observer-variation", exist_ok=True)

# ===============================================================
# INTER-OBSERVER COMBINED PLOT
# ===============================================================

fig_inter, axes_inter = plt.subplots(
    nrows=5, ncols=2, figsize=(10, 14)
)
axes_inter = axes_inter.flatten()

for i, prefix in enumerate(all_columns):
    try:
        # Use T1 comparison (KP vs KT)
        bland_altman_plot(
            axes_inter[i],
            df[f"{prefix}_11"],
            df[f"{prefix}_21"],
            f"{prefix} (T1: KP vs KT)"
        )
    except KeyError as e:
        axes_inter[i].set_title(f"{prefix} - Missing", fontsize=9)
        axes_inter[i].axis("off")

plt.tight_layout()
plt.suptitle("Bland–Altman: Inter-Observer Variation", fontsize=14, y=1.02)
plt.savefig("../figs/inter-observer-variation/bland_altman_combined_inter.png", bbox_inches='tight')
plt.close(fig_inter)

# ===============================================================
# INTRA-OBSERVER COMBINED PLOT
# ===============================================================

fig_intra, axes_intra = plt.subplots(
    nrows=5, ncols=2, figsize=(10, 14)
)
axes_intra = axes_intra.flatten()

for i, prefix in enumerate(all_columns):
    try:
        # Use KP (T1 vs T2) comparison
        bland_altman_plot(
            axes_intra[i],
            df[f"{prefix}_11"],
            df[f"{prefix}_12"],
            f"{prefix} (KP: T1 vs T2)"
        )
    except KeyError as e:
        axes_intra[i].set_title(f"{prefix} - Missing", fontsize=9)
        axes_intra[i].axis("off")

plt.tight_layout()
plt.suptitle("Bland–Altman: Intra-Observer Variation", fontsize=14, y=1.02)
plt.savefig("../figs/intra-observer-variation/bland_altman_combined_intra.png", bbox_inches='tight')
plt.close(fig_intra)

print("✅ Combined inter- and intra-observer Bland–Altman plots saved successfully!")