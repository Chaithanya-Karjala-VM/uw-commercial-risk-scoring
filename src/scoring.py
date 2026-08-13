"""
Shared scoring logic - loads the 6 trained sub-models (property/liability x frequency/
severity/large-loss) once, scores any dataframe of submissions. Both combine_scores.py and
the Streamlit app import this instead of duplicating the combination formula.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import joblib
from lob_config import PROPERTY_LOBS, group_for_lob, compute_premium

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = str(PROJECT_ROOT / "models")


def load_models():
    return {
        "frequency": joblib.load(f"{MODELS_DIR}/frequency_model.joblib"),
        "severity": joblib.load(f"{MODELS_DIR}/severity_model.joblib"),
        "large_loss": joblib.load(f"{MODELS_DIR}/large_loss_model.joblib"),
    }


def _prep_X(sub_df, cat_cols):
    X = sub_df.copy()
    for c in cat_cols:
        X[c] = X[c].astype(object).where(X[c].notna(), None)
    return X


def score(df, models, term_col="policy_term_months"):
    """
    df needs: line_of_business, policy_term_months, and every field relevant to whichever
    group (property/liability) each row's LOB belongs to (see lob_config.py). Premium is
    computed automatically from TIV (property) or employees x wage (liability) - it is not
    a required input column, though indicated_annual_premium_usd will be overwritten if
    present, so this is consistent whether scoring the training book or a brand-new submission.
    Works on 1 row or 20,000.
    """
    out = df.copy()
    out["_group"] = out["line_of_business"].apply(group_for_lob)

    has_real_premium = "indicated_annual_premium_usd" in out.columns and out["indicated_annual_premium_usd"].notna().all()

    reset_cols = ["predicted_frequency", "predicted_attritional_severity", "p_large_loss",
                  "expected_large_severity", "total_expected_loss_cost", "predicted_loss_ratio_pct",
                  "loss_propensity_score_v2"]
    if not has_real_premium:
        reset_cols.append("indicated_annual_premium_usd")
    for col in reset_cols:
        out[col] = np.nan

    for group in ["property", "liability"]:
        mask = out["_group"] == group
        if not mask.any():
            continue
        sub = out.loc[mask]

        freq_saved = models["frequency"][group]
        Xf = _prep_X(sub[freq_saved["features"]], freq_saved["cat_cols"])
        exposure_years = (sub[term_col] / 12).clip(lower=0.01)
        predicted_freq_rate = freq_saved["model"].predict(Xf)
        predicted_frequency = predicted_freq_rate * exposure_years

        sev_saved = models["severity"][group]
        Xs = _prep_X(sub[sev_saved["features"]], sev_saved["cat_cols"])
        predicted_attritional_severity = sev_saved["model"].predict(Xs)

        large_saved = models["large_loss"][group]
        Xl = _prep_X(sub[large_saved["features"]], large_saved["cat_cols"])
        p_large = large_saved["occurrence_model"].predict_proba(Xl)[:, 1]
        threshold_by_lob = sub["line_of_business"].map(models["large_loss"]["thresholds_by_lob"])
        gpd_mean_excess = stats.genpareto.mean(large_saved["gpd_shape"], scale=large_saved["gpd_scale"])
        flat_large_severity = threshold_by_lob + gpd_mean_excess

        # RISK-SENSITIVE large-loss severity: scale the flat per-LOB figure by how this
        # submission's own predicted attritional severity compares to its LOB's average -
        # ties the large-loss dollar amount to the same risk signal the severity model
        # already learned, instead of using one constant number for every submission in
        # a LOB regardless of how good or bad its own risk factors are (see conversation -
        # this was the main reason good-risk submissions weren't scoring any better).
        lob_avg = sub["line_of_business"].map(models["severity"]["lob_avg_severity"])
        severity_ratio = (predicted_attritional_severity / lob_avg).clip(0.3, 3.0)
        expected_large_severity = flat_large_severity * severity_ratio

        attritional_freq = np.clip(predicted_frequency - p_large, 0, None)
        total_expected_loss_cost = (attritional_freq * predicted_attritional_severity
                                     + p_large * expected_large_severity)

        if has_real_premium:
            premium = sub["indicated_annual_premium_usd"]
        else:
            premium = sub.apply(lambda r: compute_premium(
                r["line_of_business"],
                total_insured_value_usd=r.get("total_insured_value_usd"),
                number_of_employees=r.get("number_of_employees"),
                avg_wage_per_employee_usd=r.get("avg_wage_per_employee_usd"),
            ), axis=1)

        predicted_loss_ratio_pct = (total_expected_loss_cost / premium) * 100
        loss_propensity_score = predicted_loss_ratio_pct.clip(10, 100).round(1)

        out.loc[mask, "predicted_frequency"] = predicted_frequency.round(4)
        out.loc[mask, "predicted_attritional_severity"] = predicted_attritional_severity.round(0)
        out.loc[mask, "p_large_loss"] = p_large.round(4)
        out.loc[mask, "expected_large_severity"] = expected_large_severity.round(0)
        out.loc[mask, "total_expected_loss_cost"] = total_expected_loss_cost.round(0)
        out.loc[mask, "indicated_annual_premium_usd"] = premium.round(0)
        out.loc[mask, "predicted_loss_ratio_pct"] = predicted_loss_ratio_pct.round(1)
        out.loc[mask, "loss_propensity_score_v2"] = loss_propensity_score

    out = out.drop(columns=["_group"])
    return out
