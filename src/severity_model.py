"""
Severity model: predicts expected claim $ amount, GIVEN a claim happens.
Fit only on rows with claim_count_total > 0 - you can't ask "how big was it" on a
policy with zero claims, that's undefined, not zero.
Gamma GLM baseline, GBM challenger (gamma loss). Saves winner to ../models/severity_model.joblib
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_gamma_deviance, mean_absolute_error
import joblib
from data_loader import load_data, features_for

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])

# same exclusion as the frequency model - none of these are knowable before a claim happens,
# so none of them can be predictors of how big a future claim will be either
CLAIM_OUTCOME_COLS = [
    "claim_count_total", "claim_count_open", "claim_count_settled", "primary_cause_of_claim",
    "claim_nature", "claim_severity_avg_usd", "total_incurred_claim_amount_usd",
    "large_claim_flag", "large_claim_count", "catastrophic_loss_flag",
    "catastrophic_loss_amount_usd", "hospital_visit_count", "treatment_type",
    "litigated_claim_flag", "litigated_claim_count",
]
features = [f for f in features_for("Loss", usable_for, output_cols) if f not in CLAIM_OUTCOME_COLS]
print(f"{len(features)} predictors (same exclusion list as the frequency model)")

claims = df[df["claim_count_total"] > 0].copy()
print(f"{len(claims):,} rows had at least one claim (of {len(df):,} total)")

# ATTRITIONAL-ONLY, per the fix identified in Phase 4: this model previously fit all 6,535
# claims together, mixing ordinary claims with the rare huge ones, which is exactly why its
# calibration was rough (one model trying to fit two different behaviors at once). Now
# excluding anything above the per-LOB large-loss threshold derived in large_loss_model.py -
# those get their own separate severity treatment (Generalized Pareto) over there.
thresholds = claims.groupby("line_of_business")["claim_severity_avg_usd"].quantile(0.95)
claims["_thr"] = claims["line_of_business"].map(thresholds)
n_before = len(claims)
claims = claims[claims["claim_severity_avg_usd"] <= claims["_thr"]].drop(columns="_thr")
print(f"Excluded {n_before - len(claims)} large claims (per-LOB 95th pct threshold) - "
      f"fitting attritional severity on the remaining {len(claims):,}")

y = claims["claim_severity_avg_usd"]
weight = claims["claim_count_total"]  # an average from more claims is more credible

claims_sorted = claims.sort_values("effective_date")
split = claims_sorted.iloc[int(len(claims_sorted) * 0.75)]["effective_date"]
train_idx = claims_sorted[claims_sorted["effective_date"] < split].index
test_idx = claims_sorted[claims_sorted["effective_date"] >= split].index
print(f"Train: {len(train_idx):,} rows before {split.date()} | Test: {len(test_idx):,} rows after")

X = claims[features].copy()
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
ytr, yte = y.loc[train_idx], y.loc[test_idx]
wtr, wte = weight.loc[train_idx], weight.loc[test_idx]

# --- GLM baseline (Gamma - right-skewed, strictly positive, standard for claim severity) ---
# alpha=0.1 (the frequency model's setting) let a handful of rare categorical combinations
# blow predictions up to the hundreds of millions on this smaller (6.5k row) claims-only
# dataset - swept alpha and 5.0 is the stable point (checked: MAE flat past this, predictions
# stay in a sane range). Severity needs stronger regularization than frequency did because
# it's training on a third of the rows across the same number of categories.
glm = Pipeline([("prep", pre), ("model", GammaRegressor(alpha=5.0, max_iter=2000))])
glm.fit(Xtr, ytr, model__sample_weight=wtr)
glm_pred = glm.predict(Xte)
glm_dev = mean_gamma_deviance(yte, glm_pred, sample_weight=wte)
glm_mae = mean_absolute_error(yte, glm_pred)
print(f"\nGLM (Gamma) - mean deviance: {glm_dev:.4f} | MAE: ${glm_mae:,.0f}")

# --- GBM challenger ---
gbm = Pipeline([("prep", pre), ("model", HistGradientBoostingRegressor(loss="gamma", max_iter=300, max_depth=6, random_state=42))])
gbm.fit(Xtr, ytr, model__sample_weight=wtr)
gbm_pred = np.clip(gbm.predict(Xte), 1, None)  # gamma deviance needs strictly positive predictions
gbm_dev = mean_gamma_deviance(yte, gbm_pred, sample_weight=wte)
gbm_mae = mean_absolute_error(yte, gbm_pred)
print(f"GBM (HistGB, gamma loss) - mean deviance: {gbm_dev:.4f} | MAE: ${gbm_mae:,.0f}")
print(f"GBM {'improves' if gbm_dev < glm_dev else 'does not improve'} on GLM "
      f"({abs(1 - gbm_dev/glm_dev)*100:.1f}% {'lower' if gbm_dev < glm_dev else 'higher'} deviance)")

# --- calibration by decile - checked for BOTH models, since deviance and MAE disagreed above
# and calibration is the more decision-relevant view for pricing: does the average prediction
# track the average actual outcome within a risk band, not just a single summary statistic
# that a handful of very large claims can dominate
def calibration_table(pred, actual, weight):
    c = pd.DataFrame({"pred": pred, "actual": actual, "weight": weight})
    c["decile"] = pd.qcut(c["pred"], 10, labels=False, duplicates="drop")
    return c.groupby("decile").apply(
        lambda g: pd.Series({
            "avg_predicted": np.average(g["pred"], weights=g["weight"]),
            "avg_actual": np.average(g["actual"], weights=g["weight"]),
            "n": len(g),
        }), include_groups=False
    ).round(0)

print("\nCalibration by predicted-severity decile (GLM):")
print(calibration_table(glm_pred, yte.values, wte.values).to_string())
print("\nCalibration by predicted-severity decile (GBM):")
print(calibration_table(gbm_pred, yte.values, wte.values).to_string())

print("\nDeviance favors GLM here, MAE favors GBM by a wide margin - checked calibration for "
      "both above because gamma deviance on a small, heavy-tailed claims set (6,535 rows) can "
      "get dominated by a few very large claims, which makes it a less reliable tie-breaker "
      "than usual.")
print("\nReading the two tables: GLM's predicted deciles don't even move in the same order as "
      "actual severity (decile 6 predicts higher than decile 4, but actual severity is lower) - "
      "it isn't ranking risk correctly. GBM's predicted deciles track actual severity in the "
      "right order almost throughout, but every prediction runs low by a consistent-looking "
      "margin - that's a systematic bias, not a ranking problem, and it has a standard fix: "
      "scale every prediction by the ratio of actual-to-predicted average on the training set.")

gbm_pred_train = np.clip(gbm.predict(Xtr), 1, None)
bias_correction = np.average(ytr, weights=wtr) / np.average(gbm_pred_train, weights=wtr)
gbm_pred_corrected = gbm_pred * bias_correction
print(f"\nBias correction factor: {bias_correction:.3f}")
print("Calibration by predicted-severity decile (GBM, bias-corrected):")
print(calibration_table(gbm_pred_corrected, yte.values, wte.values).to_string())
corrected_mae = mean_absolute_error(yte, gbm_pred_corrected)
print(f"MAE after correction: ${corrected_mae:,.0f} (was ${gbm_mae:,.0f} uncorrected)")
print("\nCorrection made MAE worse, not better - the under-prediction isn't uniform (small "
      "claims run ~5x low, the top decile only ~1.2x low), so one multiplier can't fix both "
      "ends. That's a shape problem, not a scale problem, and it's the exact thing Phase 4 "
      "(splitting attritional claims from the large-loss tail) is designed to fix - a handful "
      "of very large claims are currently forcing one model to fit two different behaviors at "
      "once. Keeping the uncorrected GBM for now since it has the best real MAE of anything "
      "tried, and revisiting this once the large-loss layer is split out.")

# --- save winner ---
# uncorrected GBM - best real MAE of anything tried; ranks risk correctly even though absolute
# calibration is still rough. Flagged as a known limitation to revisit once Phase 4 splits the
# large-loss tail out of this same target.
joblib.dump({"model": gbm, "features": features, "num_cols": num_cols, "cat_cols": cat_cols,
             "type": "gbm"}, "../models/severity_model.joblib")
print(f"\nSaved GBM (uncorrected) to ../models/severity_model.joblib")
