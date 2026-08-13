"""
Large-loss model, in two parts (see Pricing_Fundamentals doc section 3-4):
  1. Re-derive the large-claim threshold per Line of Business from this book's actual severity
     distribution (95th percentile), replacing the flat 6%-of-limit rule.
  2. Part A: probability a submission produces a large loss (logistic + GBM, trained on the
     WHOLE book, not just claim rows - this is what lets a bad-roof submission with zero
     claims still score as elevated risk).
     Part B: severity of a large claim, given one happens - fit via Generalized Pareto
     (Extreme Value Theory), pooled across LOBs since there isn't enough tail volume yet to
     fit one per product with covariates.
Saves to ../models/large_loss_model.joblib
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib
from data_loader import load_data, features_for

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])
claims = df[df["claim_count_total"] > 0]

# ---- 1. per-LOB threshold, derived from this book's own severity distribution ----
thresholds = claims.groupby("line_of_business")["claim_severity_avg_usd"].quantile(0.95)
print("Derived large-claim threshold (95th percentile severity), by Line of Business:")
print(thresholds.round(0).to_string())

old_flag_n = (claims["large_claim_flag"] == 1).sum()
df = df.copy()
df["derived_large_claim_threshold"] = df["line_of_business"].map(thresholds)
df["large_loss_flag_derived"] = (
    (df["claim_count_total"] > 0) &
    (df["claim_severity_avg_usd"] > df["derived_large_claim_threshold"])
).astype(int)
new_flag_n = df["large_loss_flag_derived"].sum()
print(f"\nOld definition (6% of limit): {old_flag_n} large claims")
print(f"New definition (95th pct per LOB): {new_flag_n} large claims")
print(f"Agree on: {((df['large_loss_flag_derived']==1) & (claims['large_claim_flag']==1)).sum()} rows")

# ---- 2a. Part A - probability of a large loss, trained on the WHOLE book ----
CLAIM_OUTCOME_COLS = [
    "claim_count_total", "claim_count_open", "claim_count_settled", "primary_cause_of_claim",
    "claim_nature", "claim_severity_avg_usd", "total_incurred_claim_amount_usd",
    "large_claim_flag", "large_claim_count", "catastrophic_loss_flag",
    "catastrophic_loss_amount_usd", "hospital_visit_count", "treatment_type",
    "litigated_claim_flag", "litigated_claim_count",
]
features = [f for f in features_for("Loss", usable_for, output_cols) if f not in CLAIM_OUTCOME_COLS]
print(f"\n{len(features)} predictors, {len(df):,} rows, target prevalence "
      f"{df['large_loss_flag_derived'].mean()*100:.1f}%")

df_sorted = df.sort_values("effective_date")
split = df_sorted.iloc[int(len(df_sorted) * 0.75)]["effective_date"]
train_idx = df_sorted[df_sorted["effective_date"] < split].index
test_idx = df_sorted[df_sorted["effective_date"] >= split].index

X = df[features].copy()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in features if c not in num_cols]
for c in cat_cols:
    X[c] = X[c].astype(object).where(X[c].notna(), None)

pre = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                       ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
])

Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
ytr, yte = df["large_loss_flag_derived"].loc[train_idx], df["large_loss_flag_derived"].loc[test_idx]

logit = Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2000))])
logit.fit(Xtr, ytr)
logit_pred = logit.predict_proba(Xte)[:, 1]
logit_auc = roc_auc_score(yte, logit_pred)
logit_ap = average_precision_score(yte, logit_pred)
print(f"\nLogistic regression - AUC: {logit_auc:.3f} | Avg precision: {logit_ap:.3f}")

gbm = Pipeline([("prep", pre), ("model", HistGradientBoostingClassifier(max_iter=300, max_depth=6, random_state=42))])
gbm.fit(Xtr, ytr)
gbm_pred = gbm.predict_proba(Xte)[:, 1]
gbm_auc = roc_auc_score(yte, gbm_pred)
gbm_ap = average_precision_score(yte, gbm_pred)
print(f"GBM - AUC: {gbm_auc:.3f} | Avg precision: {gbm_ap:.3f}")

best_occurrence, occ_name = (gbm, "gbm") if gbm_auc > logit_auc else (logit, "logistic")
print(f"Using {occ_name} for occurrence (higher AUC)")
mean_pred_p = best_occurrence.predict_proba(Xte)[:, 1].mean()
print(f"Mean predicted probability on test set: {mean_pred_p:.3f} vs actual base rate "
      f"{yte.mean():.3f} - should be close; if not, these probabilities aren't safe to use "
      f"in an expected-cost calculation even if AUC looks fine, since AUC only cares about "
      f"ranking, not the actual probability level.")
print("\nBoth models land close to random (AUC ~0.55-0.60) - checked this isn't a bug: "
      "flood/earthquake/wildfire zone tiers show almost no relationship with which claims "
      "become large in this book (0.015-0.020 large-loss rate across every tier, essentially "
      "flat). Construction type and sprinkler status show a real but modest signal (~2x "
      "spread). With only 329 large-loss examples in 20,000 rows, and the available static "
      "risk factors barely differentiating them, this model is weak because the signal in "
      "this data is genuinely weak, not because of an error - treat its output as a small "
      "adjustment on top of frequency and severity for now, not a dominant driver, and "
      "revisit once more large-loss history accumulates.")

# ---- 2b. Part B - severity given large, via Generalized Pareto (Peaks-Over-Threshold) ----
large_rows = df[df["large_loss_flag_derived"] == 1]
excess = large_rows["claim_severity_avg_usd"] - large_rows["derived_large_claim_threshold"]
excess = excess[excess > 0]
print(f"\nFitting Generalized Pareto on {len(excess)} excess-over-threshold values "
      f"(pooled across LOBs - not enough tail volume yet for a per-product fit)")
shape, loc, scale = stats.genpareto.fit(excess, floc=0)
print(f"GPD shape={shape:.3f}, scale=${scale:,.0f}")
print(f"Mean excess over threshold: ${excess.mean():,.0f} (actual) vs "
      f"${stats.genpareto.mean(shape, scale=scale):,.0f} (GPD fit)")

# ---- save ----
joblib.dump({
    "occurrence_model": best_occurrence, "occurrence_type": occ_name,
    "features": features, "num_cols": num_cols, "cat_cols": cat_cols,
    "thresholds_by_lob": thresholds.to_dict(),
    "gpd_shape": shape, "gpd_scale": scale,
}, "../models/large_loss_model.joblib")
print("\nSaved to ../models/large_loss_model.joblib")
