# This script loads the raw dataset, removes duplicates and invalid angle values, and saves a cleaned version of the data for further analysis.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel file
df = pd.read_excel('../data/dataset.xlsx', engine='openpyxl')

# Basic info
print(df.info())
print(df.describe())

# Drop duplicates (example based on unique patient & foot identifier)
df = df.drop_duplicates(subset=['HN','R_1L_2'])

# Remove impossible angle values
angle_cols = [col for col in df.columns if any(x in col for x in ['cal_PA','MA','TUA','1MTA'])]
for col in angle_cols:
    df = df[(df[col] >= -30) & (df[col] <= 360)]

# Save cleaned data for next step
df.to_csv('../results/cleaned_data.csv', index=False)
print("Step 1 done: Data cleaned and saved.")