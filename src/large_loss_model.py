"""
Large-loss model, two parts, trained separately for property and liability lines.
Uses this file's OWN large_claim_flag/threshold (peril-specific: PML-based for property,
AOA-based for liability, flat $75K for WC, policy-limit-based for auto) rather than
re-deriving a generic percentile rule - this file's own definition is more sophisticated
than what we built ourselves last time.
Part A: probability of a large loss (logistic + GBM, whole book).
Part B: severity given large - Generalized Pareto, pooled per group.
No class_weight="balanced" (confirmed last time this distorts probabilities needed for
real expected-cost math).
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
from sklearn.metrics import roc_auc_score
import joblib
from data_loader import load_data
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, PROPERTY_FEATURES, LIABILITY_FEATURES, encode_ordinal_tiers

MONOTONIC_DIRECTION = {
    "financial_stability_score": -1, "claim_count_last_3_years": 1,
    "large_loss_count_last_5_years": 1, "claims_free_years": -1,
    "building_age_years": 1, "roof_age_years": 1, "fire_protection_class_iso_ppc": 1,
    "distance_to_coast_miles": -1, "distance_to_fault_line_miles": -1,
    "distance_to_nearest_water_body_miles": -1, "distance_to_fire_station_miles": 1,
    "distance_to_fire_hydrant_miles": 1, "debt_to_equity_ratio": 1,
    "flood_zone_tier": 1, "earthquake_zone_tier": 1, "wildfire_zone_tier": 1, "wind_zone_tier": 1,
}

df, schema, usable_for, output_cols = load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])
df = encode_ordinal_tiers(df)

# thresholds straight from the file's own peril-specific logic, per LOB (median of the
# file's own large_claim_threshold_usd within each line, since it varies by peril inputs)
claims_all = df[df["claim_count_total"] > 0]
thresholds = claims_all.groupby("line_of_business")["large_claim_threshold_usd"].median()
print("Per-LOB large-claim threshold (median of the file's own peril-specific value):")
print(thresholds.round(0).to_string())

GROUPS = [("property", PROPERTY_LOBS, PROPERTY_FEATURES), ("liability", LIABILITY_LOBS, LIABILITY_FEATURES)]
saved = {"thresholds_by_lob": thresholds.to_dict()}

for group_name, lobs, features in GROUPS:
    print(f"\n=== {group_name.upper()} ({', '.join(lobs)}) ===")
    sub = df[df["line_of_business"].isin(lobs)].copy()
    print(f"{len(features)} predictors, {len(sub):,} rows, target prevalence "
          f"{sub['large_claim_flag'].mean()*100:.1f}%")

    sub_sorted = sub.sort_values("effective_date")
    split = sub_sorted.iloc[int(len(sub_sorted) * 0.75)]["effective_date"]
    train_idx = sub_sorted[sub_sorted["effective_date"] < split].index
    test_idx = sub_sorted[sub_sorted["effective_date"] >= split].index

    X = sub[features].copy()
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
    ytr, yte = sub["large_claim_flag"].loc[train_idx], sub["large_claim_flag"].loc[test_idx]

    logit = Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2000))])
    logit.fit(Xtr, ytr)

    # Consistency check: a fitted coefficient that contradicts a KNOWN real-world direction
    # (the same MONOTONIC_DIRECTION list used for the GBM constraints) is a red flag on a
    # thin positive-class sample like this one (only ~120-180 large-claim rows across 14-33
    # predictors) - found exactly this on property: financial_stability_score fit at +0.10
    # (better stability -> HIGHER large-loss probability) despite a raw correlation of only
    # -0.003, essentially zero. Regularization sweep (C=1.0 down to 0.01) shrank it but never
    # flipped the sign, while degrading AUC in the process - so instead of forcing a sign or
    # accepting a wrong-direction driver, the offending feature is dropped from THIS specific
    # sub-model and it's refit, rather than either silently keeping a spurious driver or
    # discarding features that are behaving correctly in the frequency/severity models.
    ohe_check = logit.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    feat_names_check = num_cols + list(ohe_check.get_feature_names_out(cat_cols))
    coefs_check = pd.Series(logit.named_steps["model"].coef_.ravel(), index=feat_names_check)
    contradicting = [f for f, d in MONOTONIC_DIRECTION.items()
                      if f in coefs_check.index and np.sign(coefs_check[f]) not in (0, d)]
    if contradicting:
        print(f"Dropping from occurrence model - coefficient contradicts known real-world "
              f"direction (likely a thin-sample artifact, not real signal): {contradicting}")
        features_retry = [f for f in features if f not in contradicting]
        num_cols = [c for c in num_cols if c not in contradicting]
        cat_cols = [c for c in cat_cols if c not in contradicting]
        X = X[features_retry]
        pre = ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                               ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols),
        ])
        Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
        logit = Pipeline([("prep", pre), ("model", LogisticRegression(max_iter=2000))])
        logit.fit(Xtr, ytr)
        features = features_retry

    logit_pred = logit.predict_proba(Xte)[:, 1]
    logit_auc = roc_auc_score(yte, logit_pred)
    print(f"Logistic - AUC: {logit_auc:.3f}")

    pre_fitted = pre.fit(Xtr)
    n_cat_output_cols = len(pre_fitted.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
    monotonic_cst = [MONOTONIC_DIRECTION.get(c, 0) for c in num_cols] + [0] * n_cat_output_cols

    gbm = Pipeline([("prep", pre), ("model", HistGradientBoostingClassifier(
        max_iter=300, max_depth=6, random_state=42, monotonic_cst=monotonic_cst))])
    gbm.fit(Xtr, ytr)
    gbm_pred = gbm.predict_proba(Xte)[:, 1]
    gbm_auc = roc_auc_score(yte, gbm_pred)
    print(f"GBM - AUC: {gbm_auc:.3f}")

    best_occ, occ_name = (gbm, "gbm") if gbm_auc > logit_auc else (logit, "logistic")
    mean_pred_p = best_occ.predict_proba(Xte)[:, 1].mean()
    print(f"Using {occ_name}. Mean predicted probability {mean_pred_p:.3f} vs actual base rate {yte.mean():.3f}")

    large_rows = sub[sub["large_claim_flag"] == 1]
    excess = (large_rows["total_incurred_claim_amount_usd"] - large_rows["large_claim_threshold_usd"])
    excess = excess[excess > 0]
    shape, loc, scale = stats.genpareto.fit(excess, floc=0)
    print(f"GPD (n={len(excess)}): shape={shape:.3f}, scale=${scale:,.0f}")

    saved[group_name] = {
        "occurrence_model": best_occ, "occurrence_type": occ_name,
        "features": features, "num_cols": num_cols, "cat_cols": cat_cols,
        "gpd_shape": shape, "gpd_scale": scale,
    }

joblib.dump(saved, "../models/large_loss_model.joblib")
print("\nSaved to ../models/large_loss_model.joblib")
