# 🔑 Feature-Based Model Evaluation: Cluster + Importance Insights
# This script evaluates Logistic Regression (optionally Random Forest) on feature subsets ranked
# by combined cluster separation + importance (Correlation/MI/RF). It calculates classification metrics
# (precision, recall, F1-score) and AUC, then saves results to CSV for analysis.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# Load preprocessed radiographic data
df = pd.read_csv('../results/CSV/radiographic_data_preprocessed.csv')

# Features ranked by cluster separation + statistical/model importance
ranked_features = [
    'LL_mean', '1MTA_mean', 'cal_PA_mean', 'MT_calA_mean', 'ML_mean',
    'MA_mean', 'TCA_mean', 'TUA_mean', '5MTCA_mean', 'MCL_mean',
    'MCL_var', '1MTA_var', 'MT_calA_var', 'LL_var', 'cal_PA_var',
    'MA_var', 'ML_var', '5MTCA_var', 'TUA_var', 'TCA_var'
]

# Prepare feature matrix and target
X = df[ranked_features].fillna(df[ranked_features].median())
y = df['AS_0S_1']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create incremental feature subsets: top 1, top 2, ..., full set
feature_subsets = [ranked_features[:i] for i in range(1, len(ranked_features)+1)]

results = []

def evaluate_subset(features, model_type='LogisticRegression'):
    """Evaluate a model on a given subset of features and store metrics."""
    X_train_sub = X_train[features]
    X_test_sub = X_test[features]

    if model_type == 'LogisticRegression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=500, random_state=42)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train_sub, y_train)
    y_pred = model.predict(X_test_sub)
    y_proba = model.predict_proba(X_test_sub)[:, 1]

    # Compute metrics safely (handle classes with no predictions)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Features": ", ".join(features),
        "Model": model_type,
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

# Evaluate all subsets with Logistic Regression
for subset in feature_subsets:
    evaluate_subset(subset, model_type='LogisticRegression')

# Optional: Evaluate with Random Forest as well
# for subset in feature_subsets:
#     evaluate_subset(subset, model_type='RandomForest')

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('../results/model_evaluation_results.csv', index=False)

# -----------------------------
# Generate GitHub Markdown Table
# -----------------------------
def df_to_markdown(df):
    """Convert a DataFrame to a GitHub-friendly Markdown table."""
    md = df.to_markdown(index=False)
    return md

md_table = df_to_markdown(results_df)
print("\n📊 Markdown Table:\n")
print(md_table)

# Optional: Save Markdown table to file
with open('../results/model_evaluation_results.md', 'w') as f:
    f.write(md_table)

print("✅ Evaluation completed. Results saved to CSV and Markdown")