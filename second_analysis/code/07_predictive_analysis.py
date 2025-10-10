# 🔑 Feature-Based Model Evaluation: Pure Features Only
# OUTLIER cells in pure columns are ignored
# ===============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Load preprocessed radiographic data
df = pd.read_csv('../results/CSV/radiographic_data_preprocessed.csv')

# Select only pure columns
pure_features = [col for col in df.columns if 'pure' in col.lower()]
if len(pure_features) == 0:
    raise ValueError("No 'pure' columns found in dataset.")

# Feature matrix (pure columns only) and target
X = df[pure_features]
y = df['AS_0S_1']  # target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create incremental feature subsets: top 1, top 2, ..., full set
feature_subsets = [pure_features[:i] for i in range(1, len(pure_features)+1)]
results = []

def evaluate_subset(features):
    """Evaluate Logistic Regression using only pure features, ignoring OUTLIERs."""
    # Replace 'OUTLIER' with np.nan (compatible with float)
    X_train_sub = X_train[features].replace("OUTLIER", np.nan).astype(float).copy()
    X_test_sub = X_test[features].replace("OUTLIER", np.nan).astype(float).copy()

    # Drop rows that are all NaN in this subset
    X_train_sub_valid = X_train_sub.dropna(how='all')
    y_train_sub = y_train.loc[X_train_sub_valid.index]

    X_test_sub_valid = X_test_sub.dropna(how='all')
    y_test_sub = y_test.loc[X_test_sub_valid.index]

    # Skip subset if only one class remains
    if len(y_train_sub.unique()) < 2 or len(y_test_sub.unique()) < 2:
        print(f"⚠️ Skipping subset {features} due to single-class problem")
        return

    # Fill remaining NaN with train median
    X_train_sub_valid = X_train_sub_valid.fillna(X_train_sub_valid.median())
    X_test_sub_valid = X_test_sub_valid.fillna(X_train_sub_valid.median())

    # Train model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_sub_valid, y_train_sub)
    y_pred = model.predict(X_test_sub_valid)
    y_proba = model.predict_proba(X_test_sub_valid)[:, 1]

    # Metrics
    report = classification_report(y_test_sub, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test_sub, y_proba)

    results.append({
        "Features": ", ".join(features),
        "Precision (0)": round(report.get('0', {}).get('precision', 0), 3),
        "Recall (0)": round(report.get('0', {}).get('recall', 0), 3),
        "F1 (0)": round(report.get('0', {}).get('f1-score', 0), 3),
        "Precision (1)": round(report.get('1', {}).get('precision', 0), 3),
        "Recall (1)": round(report.get('1', {}).get('recall', 0), 3),
        "F1 (1)": round(report.get('1', {}).get('f1-score', 0), 3),
        "Macro F1": round(report['macro avg']['f1-score'], 3),
        "Weighted F1": round(report['weighted avg']['f1-score'], 3),
        "AUC": round(auc, 3)
    })

# Evaluate all subsets
for subset in feature_subsets:
    evaluate_subset(subset)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('../results/model_evaluation_results_pure_only.csv', index=False)

# Markdown table
md_table = results_df.to_markdown(index=False)
with open('../results/model_evaluation_results_pure_only.md', 'w') as f:
    f.write(md_table)

print("✅ Evaluation completed using only pure columns. Results saved to CSV and Markdown.")