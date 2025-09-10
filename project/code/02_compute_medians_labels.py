# This script computes median values for repeated measurements and generates clinical and radiographic AAFD labels, then saves the updated dataset.

import pandas as pd
import numpy as np

df = pd.read_csv('../results/cleaned_data.csv')

# ✅ Compute median for repeated measurements
df['cal_PA_med'] = df[['cal_PA_11','cal_PA_12','cal_PA_21','cal_PA_22']].median(axis=1)
df['MA_med']     = df[['MA_11','MA_12','MA_21','MA_22']].median(axis=1)
df['TUA_med']    = df[['TUA_11','TUA_12','TUA_21','TUA_22']].median(axis=1)
df['1MTA_med']   = df[['1MTA_11','1MTA_12','1MTA_21','1MTA_22']].median(axis=1)  # fixed underscores

# Clinical label
df['clinical_AAFD'] = df['AS_0S_1']  # 0 = no, 1 = yes

# Radiographic labels
# Label is 1 (positive) if any of the radiographic measurements are abnormal
df['rad_AAFD_A'] = (
    (df['cal_PA_med'] < 20) |
    (df['MA_med'] > 4) |
    (df['TUA_med'] > 7) |
    (df['1MTA_med'] > 12)
).astype(int)

# Score counts how many of the four are abnormal for each row (0–4)
df['rad_score'] = (
    (df['cal_PA_med'] < 20).astype(int) +
    (df['MA_med'] > 4).astype(int) +
    (df['TUA_med'] > 7).astype(int) +
    (df['1MTA_med'] > 12).astype(int)
)

# Another radiographical label: 1 if two or more abnormalities are present
df['rad_AAFD_B'] = (df['rad_score'] >= 2).astype(int)

# Save result
df.to_csv('../results/medians_labels.csv', index=False)
print("✅ Step 2 done: Medians computed and labels created.")