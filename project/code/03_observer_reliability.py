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

df = pd.read_csv('../results/medians_labels.csv')

# Reshape long for ICC - from wide format(one row with multiple columns) to long format(multiple rows with measurement and value columns)
# Reason for doing this because most stastical functional like ICC, expect data in long format
long_df = pd.melt(df, id_vars=['HN'],
                  value_vars=['cal_PA_11','cal_PA_12','cal_PA_21','cal_PA_22'],
                  var_name='measurement', value_name='value')

# Adds two columns to long_df - 'rater' which identifies who measured it(KP if first digit = 1, KT if first digit = 2), and 'timepoint' when the measurement was taken(T1 if second digit = 1, T2 if second digit = 2)
long_df['rater'] = long_df['measurement'].apply(lambda x: 'KP' if '1' in x else 'KT')
long_df['timepoint'] = long_df['measurement'].apply(lambda x: 'T1' if '1' in x.split('_')[1] else 'T2')

# ICC(Intraclass Correlation Coeficient) - used for checking the inter-rater reliability
# data - long_df, targets - HN(each patient's hosptial no), raters - measurement(which rater gave the measurement), ratings - value(the actual numeric measurement)
icc_res = pg.intraclass_corr(data=long_df, targets='HN', raters='measurement', ratings='value')
icc_res.to_csv('../results/icc_results.csv', index=False)
print(icc_res)

# Bland-Altman function (utils.py), doing it for all angles
# This is using the wide format directly, not long format
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
    if plot_type == "intra":
        folder = "../figs/intra-observer-variation"
    else:
        folder = "../figs/inter-observer-variation"

    os.makedirs(folder, exist_ok=True)
    plt.savefig(os.path.join(folder, f'bland_altman_{label}.png'))
    plt.close()
    
all_columns = [
    "cal_PA", "MA", "TCA", "ML", "LL", "MCL", 
    "MT_calA", "1MTA", "5MTCA", "TUA"
]

for prefix in all_columns:
    try:
        # Inter-observer comparisons (T1 and T2)
        bland_altman_plot(
            df[f"{prefix}_11"], df[f"{prefix}_21"],
            f"{prefix}_T1_KP_vs_KT", plot_type="inter"
        )
        bland_altman_plot(
            df[f"{prefix}_12"], df[f"{prefix}_22"],
            f"{prefix}_T2_KP_vs_KT", plot_type="inter"
        )

        # Intra-observer comparisons (KP and KT)
        bland_altman_plot(
            df[f"{prefix}_11"], df[f"{prefix}_12"],
            f"{prefix}_KP_T1_vs_T2", plot_type="intra"
        )
        bland_altman_plot(
            df[f"{prefix}_21"], df[f"{prefix}_22"],
            f"{prefix}_KT_T1_vs_T2", plot_type="intra"
        )

    except KeyError as e:
        print(f"Skipping {prefix}: missing column -> {e}")