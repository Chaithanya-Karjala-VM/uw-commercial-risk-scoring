"""
Data Audit - schema-driven, rerunnable against any future version of this file.
Run: python data_audit.py
"""
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
from data_loader import load_data

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else None
OUTPUT_REPORT = "../reports/data_audit_report.md"
REF_YEAR = pd.Timestamp.today().year

report, issues = [], []
def w(t=""):
    print(t); report.append(t)
def h(t):
    w(); w(f"## {t}"); w()

df, schema, usable_for, output_cols = load_data(INPUT_FILE) if INPUT_FILE else load_data()
df["effective_date"] = pd.to_datetime(df["effective_date"])
missing = [n for n in schema["name"] if n not in df.columns]
extra = [c for c in df.columns if c not in schema["name"].values]

w("# Data Audit Report")
w(f"Rows: {len(df):,} | Columns: {len(schema)}")

h("1. Schema")
if not missing and not extra:
    w("All columns present at expected position. No extras.")
else:
    if missing: w(f"Missing: {missing}"); issues.append("schema columns missing")
    if extra: w(f"Extra: {extra}"); issues.append("extra columns")

h("2. Missing data")
miss = (df.isna().mean() * 100).round(2)
miss = miss[miss > 0].sort_values(ascending=False)
w(miss.to_string() if len(miss) else "No missing values anywhere.")
heavy = miss[miss > 40]
if len(heavy): issues.append(f"{list(heavy.index)} missing >40%")

h("3. Internal consistency (file's own recomputed cross-checks)")
def check_pair(label, a, b, tol=0.01):
    if a not in df.columns or b not in df.columns:
        w(f"{label}: skipped, column missing"); return
    if df[a].dtype == bool or df[b].dtype == bool:
        n = (df[a].astype(bool) != df[b].astype(bool)).sum()
    else:
        n = (~np.isclose(df[a].fillna(-9e9), df[b].fillna(-9e9), atol=tol, rtol=0.001)).sum()
    w(f"{label}: {n:,} mismatches / {len(df):,} rows")
    if n: issues.append(f"{label}: {n} mismatches")

check_pair("Settled claim count vs check", "claim_count_settled", "claim_count_settled_check")
check_pair("Large-claim threshold vs check", "large_claim_threshold_usd", "large_claim_threshold_check", tol=1.0)
check_pair("Large-claim flag vs check", "large_claim_flag", "large_claim_flag_check")

n = (df["pml_usd"] > df["total_insured_value_usd"] + 1).sum()
w(f"PML > TIV violations: {n:,}")
if n: issues.append("PML exceeds TIV on some rows")

n = (df["aoy_limit_usd"] < df["aoa_limit_usd"] - 1).sum()
w(f"AOY < AOA violations: {n:,}")
if n: issues.append("AOY below AOA on some rows")

h("4. Claims-history leakage check (specifically re-verified for this file)")
has_claim = df[df["claim_count_total"] > 0]
pct_positive = (has_claim["claims_free_years"] > 0).mean() * 100
w(f"claims_free_years: {pct_positive:.1f}% of rows WITH a current claim show a positive "
  f"value (0% would mean leakage - it would be perfectly circular with the outcome). "
  f"{'PASS - genuinely backward-looking.' if pct_positive > 10 else 'FAIL - looks like leakage.'}")
if pct_positive <= 10:
    issues.append("claims_free_years still looks like leakage")

corr_3yr = df["claim_count_last_3_years"].corr(df["claim_count_total"])
w(f"corr(claim_count_last_3_years, claim_count_total) = {corr_3yr:.3f} - real signal if "
  f"positive and well below 1.0 (near 1.0 would suggest the two aren't independently sourced).")

h("5. Construction type vs. building age (previously flat/random, checked again)")
w(df.groupby("construction_type")["building_age_years"].mean().round(1).to_string())

h("6. Renewal vs. New Business - does claim frequency actually differ")
w(df.groupby("renewal_flag")["claim_count_total"].mean().round(3).to_string())

h("7. Premium bands by group (property on TIV, liability on payroll)")
from lob_config import PROPERTY_LOBS
prop = df[df["line_of_business"].isin(PROPERTY_LOBS)]
liab = df[~df["line_of_business"].isin(PROPERTY_LOBS)]
w("Property lines, premium as % of TIV:")
w((prop["indicated_annual_premium_usd"] / prop["total_insured_value_usd"] * 100).describe().round(3).to_string())
w("\nLiability lines, premium as % of payroll_revenue_usd:")
w((liab["indicated_annual_premium_usd"] / liab["payroll_revenue_usd"] * 100).describe().round(3).to_string())

h("8. Loss Propensity Score range")
w(df["loss_propensity_score_0_100"].agg(["min", "max", "mean"]).round(2).to_string())

h("9. Class balance")
for c in ["risk_tier", "loss_only_tier_after_propensity", "appetite_capacity_decision", "bind_propensity_tier"]:
    if c in df.columns:
        vc = df[c].value_counts()
        w(f"{c}: " + ", ".join(f"{k}={v:,}" for k, v in vc.items()))
        thin = vc[vc < 300]
        if len(thin): issues.append(f"{c}: {list(thin.index)} <300 rows")

h("10. Bottom line")
w(f"{len(issues)} item(s) flagged:" if issues else "No blocking issues found.")
for i in issues: w(f"- {i}")

os.makedirs("../reports", exist_ok=True)
with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\nSaved: {OUTPUT_REPORT}")
