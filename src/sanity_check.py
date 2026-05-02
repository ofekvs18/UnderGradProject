import argparse
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from utils import (
    load_data_for, load_disease_config, get_splits, compute_binary_metrics, find_youden_threshold,
    precision_at_recall_levels, ensure_dir, RESULTS_DIR,
)

parser = argparse.ArgumentParser(description="Sanity check — LR performance and formulas")
parser.add_argument("--disease", default="ra", help="Disease slug (e.g. ra, dm1)")
parser.add_argument("--split-salt", default="", help="Labeled split variant (e.g. _seed2)")
args = parser.parse_args()

# ── Setup Directories ────────────────────────────────────────────────────────
disease = load_disease_config(args.disease)

# Central directory for the master summary
BASE_SANITY_DIR = RESULTS_DIR / "sanity_check"
ensure_dir(BASE_SANITY_DIR)

# Disease-specific directory for internal logs/plots
DISEASE_DIR = BASE_SANITY_DIR / disease.name
ensure_dir(DISEASE_DIR)

# ── Helper: Formula Generator ────────────────────────────────────────────────
def get_lr_formula(clf, feature_names):
    """Generates string: logit(p) = b0 + b1*x1 + ..."""
    intercept = clf.intercept_[0]
    coeffs = clf.coef_[0]
    
    formula_parts = [f"{intercept:.4f}"]
    for coef, name in zip(coeffs, feature_names):
        sign = "+" if coef >= 0 else "-"
        formula_parts.append(f"{sign} ({abs(coef):.4f} * {name})")
    
    return "logit(p) = " + " ".join(formula_parts)

# ── Helper: Fit & Evaluate ────────────────────────────────────────────────────
def fit_evaluate(X_train, y_train, X_test, y_test, label, feature_names):
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    
    proba_test = clf.predict_proba(X_test)[:, 1]
    auc_roc = roc_auc_score(y_test, proba_test)
    auc_pr = average_precision_score(y_test, proba_test)
    
    formula = get_lr_formula(clf, feature_names)

    return {
        "model": label,
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "formula": formula
    }

# ── Execution ────────────────────────────────────────────────────────────────
df, features = load_data_for(disease.name, args.split_salt)

# Calculate Prevalence and Case Count from the ENTIRE dataset before splitting
total_cases = int(df["is_case"].sum())
total_prevalence = round(df["is_case"].mean(), 4)

train_df, test_df = get_splits(df)

# Single-feature models (Evaluating on test set)
single_results = []
for feat in features:
    tr = train_df[[feat, "is_case"]].dropna()
    te = test_df[[feat, "is_case"]].dropna()
    if tr["is_case"].sum() < 5 or te["is_case"].sum() < 5:
        continue

    res = fit_evaluate(tr[[feat]].values, tr["is_case"].values,
                       te[[feat]].values, te["is_case"].values, 
                       label=feat, feature_names=[feat])
    single_results.append(res)

# All-features model (Evaluating on test set)
tr_all = train_df[features + ["is_case"]].dropna()
te_all = test_df[features + ["is_case"]].dropna()
res_all = fit_evaluate(tr_all[features].values, tr_all["is_case"].values,
                       te_all[features].values, te_all["is_case"].values, 
                       label="all_features", feature_names=features)

# ── Master Summary Logic ─────────────────────────────────────────────────────
# Select best single feature based on AUC-PR
best_res_single = max(single_results, key=lambda x: x["auc_pr"])

new_row = {
    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Disease": disease.name,
    "Split_Salt": args.split_salt,
    "Total_Case_Count": total_cases,
    "Total_Prevalence": total_prevalence,
    "Best_Single_Feat": best_res_single["model"],
    "Best_Single_AUC_PR": best_res_single["auc_pr"],
    "Best_Single_AUC_ROC": best_res_single["auc_roc"],
    "Best_Single_Formula": best_res_single["formula"],
    "All_Feat_AUC_PR": res_all["auc_pr"],
    "All_Feat_AUC_ROC": res_all["auc_roc"],
    "All_Feat_Formula": res_all["formula"]
}

master_csv_path = BASE_SANITY_DIR / "master_sanity_summary.csv"

if master_csv_path.exists():
    master_df = pd.concat([pd.read_csv(master_csv_path), pd.DataFrame([new_row])], ignore_index=True)
else:
    master_df = pd.DataFrame([new_row])

master_df.to_csv(master_csv_path, index=False)

print(f"Updated master summary for {disease.name} at: {master_csv_path}")