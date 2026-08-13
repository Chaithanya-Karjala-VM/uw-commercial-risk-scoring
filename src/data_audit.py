"""
Data Audit
Reads schema from the Data Dictionary tab (column letter + usable-for tags), so this
re-runs correctly against any future version of the file with the same layout.
Run: python data_audit.py ../data/Copy_of_UW_Risk_Score_Commercial_data.xlsx
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats
from data_loader import load_data

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "../data/Copy_of_UW_Risk_Score_Commercial_data.xlsx"
OUTPUT_REPORT = "../reports/data_audit_report.md"
REF_YEAR = pd.Timestamp.today().year

report, issues = [], []
def w(t=""):
    print(t); report.append(t)
def h(t):
    w(); w(f"## {t}"); w()

df, schema, usable_for, output_cols = load_data(INPUT_FILE)
df_raw_cols_n = len(pd.read_excel(INPUT_FILE, sheet_name="Sample Data", nrows=0).columns)
missing = [n for n in schema["name"] if n not in df.columns]
extra = [c for c in df.columns if c not in schema["name"].values]

w("# Data Audit Report")
w(f"File: {INPUT_FILE} | Rows: {len(df):,} | Columns: {df_raw_cols_n}")

# 1. schema check
h("1. Schema")
if not missing and not extra:
    w("All 144 dictionary columns present at expected position. No extra columns.")
else:
    if missing: w(f"Missing from data: {missing}"); issues.append("schema columns missing")
    if extra: w(f"Extra in data, not in dictionary: {extra}"); issues.append("extra columns")

# 2. missingness
h("2. Missing data")
miss = (df.isna().mean() * 100).round(2)
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss) == 0:
    w("No missing values anywhere.")
else:
    w(miss.to_string())
    heavy = miss[miss > 40]
    if len(heavy):
        issues.append(f"{list(heavy.index)} missing >40%")

# 3. distributions
h("3. Distributions")
numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Submission ID"]
count_like, continuous = [], []
for c in numeric_cols:
    s = df[c].dropna()
    if len(s) < 10: continue
    is_int = np.allclose(s % 1, 0, atol=1e-6)
    (count_like if is_int and s.nunique() < 30 and s.min() >= 0 else continuous).append(c)

w("Continuous fields - mean, skew, % outliers beyond 3 s.d.:")
rows_ = []
flagged = []
for c in continuous:
    s = df[c].dropna()
    if s.std(ddof=0) == 0: continue
    z = (s - s.mean()) / s.std(ddof=0)
    pct = (z.abs() > 3).mean() * 100
    rows_.append((c, round(s.mean(), 1), round(stats.skew(s), 2), round(pct, 2)))
    if pct > 5: flagged.append(c)
w(pd.DataFrame(rows_, columns=["column", "mean", "skew", "pct_outliers"]).to_string(index=False))
if flagged: issues.append(f"{flagged} outlier rate >5%")

w("\nCount/gated fields - % at zero, % positive:")
rows_ = [(c, round((df[c].dropna()==0).mean()*100,1), round((df[c].dropna()>0).mean()*100,1))
         for c in count_like]
w(pd.DataFrame(rows_, columns=["column", "pct_zero", "pct_positive"]).to_string(index=False))

# 4. internal consistency
h("4. Internal consistency (file's own recomputed cross-checks)")
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

if "pml_usd" in df.columns:
    n = (df["pml_usd"] > df["total_insured_value_usd"] + 1).sum()
    w(f"PML > TIV violations: {n:,}")
    if n: issues.append("PML exceeds TIV on some rows")

if "aoa_limit_usd" in df.columns:
    n = (df["aoy_limit_usd"] < df["aoa_limit_usd"] - 1).sum()
    w(f"AOY < AOA violations: {n:,}")
    if n: issues.append("AOY below AOA on some rows")

if "year_built" in df.columns:
    off = ((REF_YEAR - df["year_built"] - df["building_age_years"]).abs() > 1).sum()
    w(f"Building age vs year built, off by >1yr: {off:,}")

# UW score reconstruction
appetite_c, bind_c, loss_c, uw_c = "appetite_capacity_score_0_100", "bind_propensity_score_0_100", \
    "loss_propensity_score_0_100", "final_priority_score_0_100"
if all(c in df.columns for c in [appetite_c, bind_c, loss_c, uw_c]):
    recon = 0.30*df[appetite_c] + 0.20*df[bind_c] + 0.50*(100-df[loss_c])
    diff = (recon - df[uw_c]).abs()
    w(f"UW Score reconstruction: avg diff {diff.mean():.2f}, {(diff<1).mean()*100:.1f}% within 1pt")
    if diff.mean() > 0.5: issues.append("UW score formula mismatch")

# 5. sub-score independence
h("5. Sub-score correlations (should be low - independent questions)")
if all(c in df.columns for c in [appetite_c, bind_c, loss_c]):
    w(f"Loss vs Bind: {df[loss_c].corr(df[bind_c]):.3f}")
    w(f"Loss vs Appetite: {df[loss_c].corr(df[appetite_c]):.3f}")
    w(f"Bind vs Appetite: {df[bind_c].corr(df[appetite_c]):.3f}")

# 6. feature inventory per model
h("6. Candidate features per model target")
w(f"Output columns (never use as features): {len(output_cols)}")
for target in ["Loss", "Bind", "Appetite", "Broker Segmentation"]:
    feats = [n for n, u in usable_for.items() if isinstance(u, str) and target in u and n not in output_cols]
    w(f"{target}: {len(feats)} features, {len(df)/max(len(feats),1):.0f} rows/feature")

# 7. class balance
h("7. Class balance")
for c in ["risk_tier", "loss_only_tier_after_propensity", "appetite_capacity_decision", "bind_propensity_tier"]:
    if c in df.columns:
        vc = df[c].value_counts()
        w(f"{c}: " + ", ".join(f"{k}={v:,}" for k, v in vc.items()))
        thin = vc[vc < 300]
        if len(thin): issues.append(f"{c}: {list(thin.index)} <300 rows")

# 8. bottom line
h("8. Bottom line")
w(f"{len(issues)} item(s) flagged:" if issues else "No blocking issues found.")
for i in issues: w(f"- {i}")

with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(report))
print(f"\nSaved: {OUTPUT_REPORT}")
