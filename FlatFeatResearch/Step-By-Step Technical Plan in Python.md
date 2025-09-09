### Step 0: Setup — files & environment

1. **Folder structure**:
```
project/
 ├─ data/          # raw CSV/Excel files
 ├─ code/          # Python scripts / notebooks
 ├─ results/       # tables, stats outputs
 └─ figs/          # plots, diagrams
```
2. **Python packages to install**:
```
pip install pandas numpy scipy statsmodels scikit-learn pingouin matplotlib seaborn bootstrapped
```
3. **Import libraries** at the top of your script:
```
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
```

### Step 1: Load & clean data

1. Load your dataset:
```
df = pd.read_csv('data/dataset.csv')
```
2. Check for missing values, duplicates, or impossible values:
```
print(df.info())
print(df.describe())
df = df.drop_duplicates(subset=['HN','R_1L_2'])
```
3. Remove impossible angle values (example: < -30° or > 360°):
```
angle_cols = [col for col in df.columns if 'cal_PA' in col or 'MA' in col or 'TUA' in col or '1MTA' in col]
for col in angle_cols:
    df = df[(df[col]>=-30) & (df[col]<=360)]
```

### Step 2: Compute representative values (medians)

For each foot, take the median across repeated measurements:
```
df['cal_PA_med'] =  df[['cal_PA_11','cal_PA_12','cal_PA_21','cal_PA_22']].median(axis=1)
df['MA_med'] = df[['MA_11','MA_12','MA_21','MA_22']].median(axis=1)
df['TUA_med'] = df[['TUA_11','TUA_12','TUA_21','TUA_22']].median(axis=1)
df['1MTA_med'] = df[['1MTA_11','1MTA_12','1MTA_21','1MTA_22']].median(axis=1)
```

### Step 3: Create labels( Labels are things we are trying to predict)

1. **Clinical label** (from your dataset directly):
```
df['clinical_AAFD'] = df['AS_0S_1']  # 0 = no, 1 = yes
```
2. **Radiographic labels** (two rules):
```
# Rule A: sensitive
df['rad_AAFD_A'] = ((df['cal_PA_med'] < 20) | 
                     (df['MA_med'] > 4) | 
                     (df['TUA_med'] > 7) | 
                     (df['1MTA_med'] > 12)).astype(int)

# Rule B: composite / stricter
df['rad_score'] = ((df['cal_PA_med'] < 20).astype(int) + 
                   (df['MA_med'] > 4).astype(int) + 
                   (df['TUA_med'] > 7).astype(int) + 
                   (df['1MTA_med'] > 12).astype(int))
df['rad_AAFD_B'] = (df['rad_score'] >= 2).astype(int)
```

### Step 4: Observer reliability (ICC & Bland–Altman)

1. Reshape data to long format for pingouin ICC. pingouin is a Python library for statistical analysis:
```
long_df = pd.melt(df, id_vars=['HN'], 
                  value_vars=['cal_PA_11','cal_PA_12','cal_PA_21','cal_PA_22'],
                  var_name='measurement', value_name='value')
# Add columns 'rater' and 'timepoint' from measurement names
long_df['rater'] = long_df['measurement'].apply(lambda x: 'KP' if '1' in x else 'KT')
long_df['timepoint'] = long_df['measurement'].apply(lambda x: 'T1' if '1' in x.split('_')[1] else 'T2')
```
2. Compute ICC (example: two-way mixed, absolute agreement):
```
icc_res = pg.intraclass_corr(data=long_df, targets='HN', raters='measurement', ratings='value')
print(icc_res)
```
3. **Bland–Altman plots**(shows difference between the measurements):
```
def bland_altman_plot(measure1, measure2, label):
    mean = (measure1 + measure2) / 2
    diff = measure1 - measure2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    plt.scatter(mean, diff)
    plt.axhline(md, color='red')
    plt.axhline(md + 1.96*sd, color='blue', linestyle='--')
    plt.axhline(md - 1.96*sd, color='blue', linestyle='--')
    plt.title(f'Bland-Altman: {label}')
    plt.show()
```

### Step 5: Compute deviations & standardize

First, measure **how abnormal each angle is**, then convert these into **standardized scores** so you can compare them fairly across angles.

1. Compute deviation from abnormal threshold:
```
df['dev_calPA'] = np.maximum(0, 20 - df['cal_PA_med'])
df['dev_MA'] = np.maximum(0, df['MA_med'] - 4)
df['dev_TUA'] = np.maximum(0, df['TUA_med'] - 7)
df['dev_1MTA'] = np.maximum(0, df['1MTA_med'] - 12)
```
2. Standardize deviations (Z-score):
```
scaler = StandardScaler()
df[['z_calPA','z_MA','z_TUA','z_1MTA']] = scaler.fit_transform(df[['dev_calPA','dev_MA','dev_TUA','dev_1MTA']])
```

### Step 6: Exploratory & univariate analysis

1. Plots: 
```
sns.histplot(df['cal_PA_med'], kde=True)
sns.boxplot(x='clinical_AAFD', y='MA_med', data=df)
sns.heatmap(df[['cal_PA_med','MA_med','TUA_med','1MTA_med']].corr(), annot=True)
```
2. **Compare symptomatic vs asymptomatic**:
```
for col in ['cal_PA_med','MA_med','TUA_med','1MTA_med']:
    stat, p = stats.mannwhitneyu(df.loc[df['clinical_AAFD']==0, col],
                                 df.loc[df['clinical_AAFD']==1, col])
    print(f'{col}: p={p}')
```
3. **ROC / AUC**:
```
from sklearn.metrics import roc_auc_score, roc_curve

for col in ['z_calPA','z_MA','z_TUA','z_1MTA']:
    auc = roc_auc_score(df['clinical_AAFD'], df[col])
    print(f'{col} AUC={auc:.2f}')
```

### Step 7: Multivariable modeling

1. **Select reliable predictors** (ICC > 0.6):
```
X = df[['z_calPA','z_MA']]  # example top ICC variables
y = df['clinical_AAFD']
```
2. **Logistic regression**:
```
model = LogisticRegression(class_weight='balanced', solver='liblinear')
model.fit(X, y)
print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
```
3. **Cross-validation (LOOCV)**:
```
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
aucs = []

for train_index, test_index in loo.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    y_pred_prob = model.predict_proba(X_test)[:,1]
    aucs.append(roc_auc_score(y_test, y_pred_prob))

print('LOOCV mean AUC:', np.mean(aucs))
```

### Step 8: Dominant parameter rule

1. Find the highest standardized deviation per foot:
```
df['dominant_var'] = df[['z_calPA','z_MA','z_TUA','z_1MTA']].idxmax(axis=1)
df['dominant_score'] = df[['z_calPA','z_MA','z_TUA','z_1MTA']].max(axis=1)
df['dominant_AAFD'] = (df['dominant_score'] > 1.5).astype(int)  # threshold example
```
2. Compare to labels:
```
print(confusion_matrix(df['rad_AAFD_A'], df['dominant_AAFD']))
print(confusion_matrix(df['clinical_AAFD'], df['dominant_AAFD']))
```

### Step 9: Save outputs

```
df.to_csv('results/cleaned_dataset_with_labels.csv', index=False)
```
