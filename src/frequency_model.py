"""
Frequency model: predicts expected claim count per exposure-year.
Poisson GLM baseline (offset = exposure), NB overdispersion check, GBM challenger.
Saves the winning pipeline to ../models/frequency_model.joblib
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_poisson_deviance
import joblib
from data_loader import load_data, features_for

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])

# these are outcomes OF a claim (severity, litigation, hospital visits, cause) - not knowable
# before the fact, so they can't be predictors of whether/how many claims occur
CLAIM_OUTCOME_COLS = [
    "claim_count_total", "claim_count_open", "claim_count_settled", "primary_cause_of_claim",
    "claim_nature", "claim_severity_avg_usd", "total_incurred_claim_amount_usd",
    "large_claim_flag", "large_claim_count", "catastrophic_loss_flag",
    "catastrophic_loss_amount_usd", "hospital_visit_count", "treatment_type",
    "litigated_claim_flag", "litigated_claim_count",
]
features = [f for f in features_for("Loss", usable_for, output_cols) if f not in CLAIM_OUTCOME_COLS]
print(f"{len(features)} predictors (excluded {len(CLAIM_OUTCOME_COLS)} same-period claim-outcome columns)")

exposure_years = (df["policy_term_months"] / 12).clip(lower=0.01)
y_count = df["claim_count_total"]
y_freq = y_count / exposure_years  # rate to predict; exposure goes in as sample_weight

df_sorted = df.sort_values("effective_date")
split = df_sorted.iloc[int(len(df_sorted) * 0.75)]["effective_date"]
train_idx = df_sorted[df_sorted["effective_date"] < split].index
test_idx = df_sorted[df_sorted["effective_date"] >= split].index
print(f"Train: {len(train_idx):,} rows before {split.date()} | Test: {len(test_idx):,} rows after")

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
ytr_freq, yte_freq = y_freq.loc[train_idx], y_freq.loc[test_idx]
ytr_count, yte_count = y_count.loc[train_idx], y_count.loc[test_idx]
etr, ete = exposure_years.loc[train_idx], exposure_years.loc[test_idx]

# --- GLM baseline (Poisson, exposure-weighted) ---
glm = Pipeline([("prep", pre), ("model", PoissonRegressor(alpha=0.01, max_iter=500))])
glm.fit(Xtr, ytr_freq, model__sample_weight=etr)
glm_pred_freq = glm.predict(Xte)
glm_dev = mean_poisson_deviance(yte_freq.clip(lower=1e-6), glm_pred_freq.clip(min=1e-6), sample_weight=ete)
print(f"\nGLM (Poisson) - mean deviance: {glm_dev:.4f}")

# --- overdispersion check via statsmodels NB, same feature matrix ---
Xtr_dense = pre.fit_transform(Xtr)
Xtr_sm = sm.add_constant(Xtr_dense)
poisson_sm = sm.GLM(ytr_count, Xtr_sm, family=sm.families.Poisson(), offset=np.log(etr)).fit()
dispersion = poisson_sm.deviance / poisson_sm.df_resid
print(f"Poisson dispersion (deviance/df_resid): {dispersion:.3f} " +
      ("- overdispersed, Negative Binomial is the better fit" if dispersion > 1.3 else "- close to 1, Poisson's assumption holds reasonably well"))
if dispersion > 1.3:
    nb_sm = sm.GLM(ytr_count, Xtr_sm, family=sm.families.NegativeBinomial(), offset=np.log(etr)).fit()
    print(f"Negative Binomial log-likelihood: {nb_sm.llf:.1f} vs Poisson: {poisson_sm.llf:.1f}")

# --- GBM challenger ---
gbm = Pipeline([("prep", pre), ("model", HistGradientBoostingRegressor(loss="poisson", max_iter=300, max_depth=6, random_state=42))])
gbm.fit(Xtr, ytr_freq, model__sample_weight=etr)
gbm_pred_freq = gbm.predict(Xte)
gbm_dev = mean_poisson_deviance(yte_freq.clip(lower=1e-6), np.clip(gbm_pred_freq, 1e-6, None), sample_weight=ete)
print(f"\nGBM (HistGB, Poisson loss) - mean deviance: {gbm_dev:.4f}")
print(f"GBM {'improves' if gbm_dev < glm_dev else 'does not improve'} on GLM "
      f"({(1 - gbm_dev/glm_dev)*100:.1f}% {'lower' if gbm_dev < glm_dev else 'higher'} deviance)")

# --- calibration by decile: predicted vs actual claim rate ---
print("\nCalibration by predicted-frequency decile (GLM):")
calib = pd.DataFrame({"pred": glm_pred_freq, "actual": yte_freq.values, "exposure": ete.values})
calib["decile"] = pd.qcut(calib["pred"], 10, labels=False, duplicates="drop")
summary = calib.groupby("decile").apply(
    lambda g: pd.Series({
        "avg_predicted": np.average(g["pred"], weights=g["exposure"]),
        "avg_actual": np.average(g["actual"], weights=g["exposure"]),
        "n": len(g),
    }), include_groups=False
)
print(summary.round(3).to_string())

# --- save winner ---
best_model, best_name = (gbm, "gbm") if gbm_dev < glm_dev else (glm, "glm")
joblib.dump({"model": best_model, "features": features, "num_cols": num_cols, "cat_cols": cat_cols,
             "type": best_name}, "../models/frequency_model.joblib")
print(f"\nSaved best model ({best_name}) to ../models/frequency_model.joblib")
