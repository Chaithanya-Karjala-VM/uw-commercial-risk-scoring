"""
Appetite/Capacity - rules-based, not trained (same reasoning as bind_propensity.py: no
historical outcome exists to train against, this is a policy decision with a known
structure, not a pattern to learn).

Hard-rule veto: the 4 named flags (sanctions, insolvency, restricted ops, region x LOB) are
a confirmed, exact NECESSARY condition - any one True means 0% pass, verified with zero
exceptions across all 20,000 rows. They are NOT sufficient - among rows where all 4 are
clear, only 70.4% actually pass. Tested TIV thresholds, Line of Business, and CAT zone
combinations as possible explanations; none produced a clean deterministic rule. Open item,
not resolved - see project notes.

The CAT-zone + TIV interaction is approximated as a soft penalty on the fit score, not a
hard veto, for the same reason - no clean threshold found despite testing.

Concentration cap is confirmed to be a flat 15% for every industry (checked directly - zero
variation across all 10 industry classes), not a per-industry lookup table.

Per manager direction: State and Coverage are now real inputs.
  - State: reinsurance_binding_headroom_pct correlates with the real Appetite/Capacity score
    at 0.66 - genuinely strong, unlike the earlier attempt to add State as a plain regression
    dummy (which added nothing). It varies by more than just state (individual submission CAT
    exposure affects it too), so a brand-new submission uses the real per-state average as a
    lookup - not an exact reproduction of the original per-row calculation, but a solid proxy.
    Combined with industry concentration headroom using "tightest constraint governs" - the
    same logic the source file itself uses to blend state/regional/country reinsurance tiers.
  - Coverage (coverage_limit_requested_usd): correlates at 0.012 - negligible, included
    honestly with a small weight rather than left out or artificially inflated.

submission_completeness_pct is no longer collected as a live input (removed from the
Submission form per manager direction). It barely varies across the book (mean sits at
90-92% regardless of segment) so a fixed constant is used here instead of a per-submission
value - see BOOK_AVG_COMPLETENESS below.
"""
import numpy as np
import pandas as pd

CONCENTRATION_CAP_PCT = 0.15  # flat cap, confirmed identical across every industry class
BOOK_AVG_COMPLETENESS = 91.4  # submission_completeness_pct barely varies across the real book

BROKER_TIER_MAP = {"Platinum": 4, "Gold": 3, "Silver": 2, "Bronze": 1}

# real per-state average reinsurance binding headroom - negative means that state is already
# over its treaty capacity on average (Oklahoma and Washington stand out - real CAT exposure
# story, not noise)
STATE_HEADROOM_AVERAGES = {
    'AL': 0.274, 'AZ': 0.369, 'CA': 0.609, 'CO': 0.312, 'CT': 0.258, 'FL': -0.596, 'GA': 0.125,
    'ID': 0.305, 'IL': 0.621, 'IN': 0.696, 'LA': -0.040, 'MA': 0.187, 'MI': 0.837, 'MO': 0.395,
    'MS': 0.137, 'MT': 0.341, 'NC': 0.022, 'NJ': -0.172, 'NM': 0.467, 'NV': 0.253, 'NY': 0.323,
    'OH': 0.861, 'OK': -2.082, 'OR': 0.290, 'PA': 0.572, 'SC': -0.096, 'TX': 0.121, 'UT': 0.440,
    'WA': -1.715, 'WI': 0.876, 'WY': 0.409,
}
NATIONAL_AVG_HEADROOM = 0.20  # fallback for a state not in the lookup

# calibrated against each measure's own real range (see compute_capacity_headroom's comment) -
# industry concentration spans roughly -1.4 to +9.1 percentage points across the 10 real
# classes; state reinsurance headroom spans roughly -208 to +88 (Oklahoma and Washington are
# structurally, severely over treaty capacity - a real CAT-exposure story, not noise). One
# shared multiplier can't be well-calibrated for both at once.
INDUSTRY_HEADROOM_SCALE = 3.0
STATE_HEADROOM_SCALE = 0.25

# real per-industry share of the book's total TIV, computed once from the actual 20,000-row
# book (python -c "from data_loader import load_data; df,*_ = load_data(); print((df.groupby('industry_class_naics')['total_insured_value_usd'].sum() / df['total_insured_value_usd'].sum()).round(4))")
# - Manufacturing already sits slightly ABOVE the 15% concentration cap (16.36%), a real,
# mild capacity constraint. Regenerate this dict (same command) whenever the real book's
# industry mix changes materially.
#
# THE BUG THIS REPLACES: compute_capacity_headroom() used to recompute concentration from
# whatever dataframe was passed into score() itself. That's correct when scoring the full
# 20,000-row book (combine_scores.py, the dashboard) - the batch IS the book. But every
# single live Submission-tab score (and every test in test_model_behavior.py) calls score()
# on a ONE-ROW dataframe - so that one row's own industry was silently treated as 100% of
# the reference book, every time, for every industry. That forced the headroom component to
# its floor of 0 on almost every live submission, regardless of the real book's actual
# exposure - found 2026-08 while walking a brand-new submission through the pipeline by hand
# and noticing the appetite score didn't move the way the real concentration numbers implied
# it should. Using this fixed, real reference instead means a single submission's score no
# longer depends on what else happens to be in the same batch it was scored alongside.
INDUSTRY_CONCENTRATION_BY_NAICS = {
    "Accommodation & Food Services": 0.0772, "Administrative Services": 0.0588,
    "Construction": 0.1355, "Health Care": 0.0732, "Manufacturing": 0.1636,
    "Professional Services": 0.0828, "Real Estate": 0.0942, "Retail Trade": 0.1224,
    "Transportation & Warehousing": 0.0898, "Wholesale Trade": 0.1025,
}


def hard_rule_pass(df):
    """Exact, verified deterministic veto - any one of these 4 flags true means automatic fail."""
    return (
        (df["sanctions_ofac_watchlist_flag"] == 0)
        & (df["prior_insolvency_bankruptcy_flag"] == 0)
        & (df["restricted_operations_flag"] == 0)
        & (df["region_lob_restricted_flag"] == 0)
    )


def _is_severe(series):
    """flood/earthquake/wildfire zone tier arrives as the string "Severe" when called on raw,
    unencoded data (this module's own __main__ verification block below) but as the ordinal
    integer 3 when called from the real pipeline (scoring.py encodes Low/Moderate/High/Severe
    -> 0-3 BEFORE calling compute_appetite_capacity_score, for the loss models' benefit).
    Comparing only against the string meant this penalty compared "3" == "Severe" in the real
    pipeline - always False - and silently never fired for a single live-scored submission,
    found 2026-08 by scoring an identical submission with and without a severe CAT zone and
    seeing zero difference in the appetite score. Handles both representations so it actually
    fires wherever it's called from."""
    # dtype == object used to be a safe "is this still text" check, but pandas 3.x's default
    # string columns (what data_loader.load_data() actually returns) use a newer Arrow-backed
    # string dtype that ISN'T object - dtype==object silently missed it, fell through to the
    # numeric branch, and crashed comparing text to 3. is_numeric_dtype is robust to whichever
    # string backend pandas is using.
    if pd.api.types.is_numeric_dtype(series):
        return series >= 3
    return series == "Severe"


def cat_tiv_penalty(df):
    """Approximated soft penalty, NOT an exact reconstruction - see module docstring."""
    any_severe_cat = (
        _is_severe(df["flood_zone_tier"]) | _is_severe(df["earthquake_zone_tier"])
        | _is_severe(df["wildfire_zone_tier"])
    )
    high_tiv = df["total_insured_value_usd"] > 2_000_000
    return np.where(any_severe_cat & high_tiv, -12, np.where(any_severe_cat, -8, 0))


def compute_appetite_fit(df, coefficients):
    passed = hard_rule_pass(df)
    broker_tier_num = df["broker_tier"].map(BROKER_TIER_MAP).fillna(2)
    completeness = df["submission_completeness_pct"] if "submission_completeness_pct" in df.columns \
        else pd.Series(BOOK_AVG_COMPLETENESS, index=df.index)
    coverage = df["coverage_limit_requested_usd"].fillna(0) if "coverage_limit_requested_usd" in df.columns \
        else pd.Series(0, index=df.index)
    score = pd.Series(coefficients["intercept"], index=df.index)
    score = score + passed.astype(float) * coefficients["appetite_hard_rule_pass_flag"]
    score = score + df["target_growth_segment_flag"].fillna(0) * coefficients["target_growth_segment_flag"]
    score = score + df["financial_stability_score"].fillna(0) * coefficients["financial_stability_score"]
    score = score + broker_tier_num * coefficients["broker_tier_num"]
    score = score + completeness.fillna(BOOK_AVG_COMPLETENESS) * coefficients["submission_completeness_pct"]
    score = score + coverage * coefficients.get("coverage_limit_requested_usd", 0.0)
    score = score + cat_tiv_penalty(df)
    score = np.where(passed, score.clip(0, 100), 0)  # hard rule failure forces 0 regardless of fit
    return np.round(score, 1)


def compute_capacity_headroom(df, book_for_concentration=None):
    """Blends industry concentration headroom with state-level reinsurance headroom, taking
    the tighter (more restrictive) of the two - same "tightest constraint governs" logic the
    source file itself uses across its own state/regional/country reinsurance tiers.

    Industry concentration is measured against the REAL book (INDUSTRY_CONCENTRATION_BY_NAICS)
    by default, not against whatever df happens to be passed in - see the module-level
    constant's comment for why. Pass book_for_concentration explicitly only if you genuinely
    want concentration measured against a different reference portfolio (e.g. a full re-score
    of a brand-new book during a retrain, where the new book itself IS the right reference)."""
    if book_for_concentration is not None:
        reference = book_for_concentration
        concentration_by_industry = (
            reference.groupby("industry_class_naics")["total_insured_value_usd"].sum()
            / reference["total_insured_value_usd"].sum()
        )
    else:
        concentration_by_industry = pd.Series(INDUSTRY_CONCENTRATION_BY_NAICS)
    concentration = df["industry_class_naics"].map(concentration_by_industry).fillna(0)
    industry_headroom_pct = (CONCENTRATION_CAP_PCT - concentration) * 100

    if "risk_state" in df.columns:
        state_headroom_pct = df["risk_state"].map(STATE_HEADROOM_AVERAGES).fillna(NATIONAL_AVG_HEADROOM) * 100
    else:
        state_headroom_pct = pd.Series(NATIONAL_AVG_HEADROOM * 100, index=df.index)

    # Scaled to a comparable 0-100 range HERE, each on its own real span, before combining -
    # found 2026-08 that combining the two raw percentage-point values first and applying one
    # shared multiplier afterward meant a constant tuned for industry's narrow real range
    # (-1.4 to +9.1 pts) made state's much wider real range (-208 to +88 pts) saturate almost
    # immediately: 4 of 31 states (FL, NJ, OK, WA) all floored to the identical score despite
    # genuinely different degrees of capacity constraint. "Tightest constraint governs" only
    # means something if both sides are on the same scale first.
    industry_component = np.clip(50 + industry_headroom_pct * INDUSTRY_HEADROOM_SCALE, 0, 100)
    state_component = np.clip(50 + state_headroom_pct * STATE_HEADROOM_SCALE, 0, 100)
    return np.minimum(industry_component, state_component)


def compute_appetite_capacity_score(df, coefficients, book_for_concentration=None):
    fit = compute_appetite_fit(df, coefficients)
    headroom_component = compute_capacity_headroom(df, book_for_concentration)  # already 0-100 scaled
    passed = hard_rule_pass(df)
    score = np.where(passed, np.clip(0.75 * fit + 0.25 * headroom_component, 0, 100), 0)
    return np.round(score, 1)


# thresholds calibrated against the real book's own fit/headroom distributions among rows that
# pass the hard-rule gate (fit: mean 54.9, IQR 50.9-58.8, unchanged by the headroom rescaling
# above). headroom: re-calibrated 2026-08 after the industry/state rescaling fix changed its
# distribution from mean 56.4/IQR 43.2-80.1 to mean 50.4/IQR 45.9-58.3 - the old thresholds
# (40/60) would have sat almost entirely outside the new, much less saturated range (60 was
# ABOVE the new 75th percentile of 58.3, which would have made "Within Appetite" nearly
# unreachable). Not a reconstruction of the original file's own 4-category split (that logic
# isn't recoverable from the fields available - same limitation noted throughout this module) -
# a defensible, real classification built on the two components this formula actually computes.
FIT_MARGINAL_THRESHOLD = 50    # below the real book's ~25th percentile fit among passing rows
FIT_STRONG_THRESHOLD = 58      # at/above the real book's ~75th percentile
HEADROOM_CONSTRAINED_THRESHOLD = 46  # below the real book's ~25th percentile headroom
HEADROOM_COMFORTABLE_THRESHOLD = 55  # at/above the real book's median


def compute_appetite_decision(df, coefficients, book_for_concentration=None):
    """Categorizes each row the same way the source file's own appetite_capacity_decision
    column does conceptually (Decline / Refer-Marginal Fit / Refer-Capacity Constrained /
    Within Appetite), but derived from THIS formula's own fit and headroom components instead
    of reverse-engineering the original's unrecoverable category logic."""
    fit = pd.Series(compute_appetite_fit(df, coefficients), index=df.index)
    headroom_component = pd.Series(compute_capacity_headroom(df, book_for_concentration), index=df.index)
    passed = hard_rule_pass(df)

    decision = pd.Series("Refer – Marginal Fit", index=df.index)  # default/catch-all
    decision[(fit >= FIT_STRONG_THRESHOLD) & (headroom_component >= HEADROOM_COMFORTABLE_THRESHOLD)] = "Within Appetite"
    decision[headroom_component < HEADROOM_CONSTRAINED_THRESHOLD] = "Refer – Capacity Constrained"
    decision[fit < FIT_MARGINAL_THRESHOLD] = "Refer – Marginal Fit"  # weak fit overrides a capacity read
    decision[~passed] = "Decline – Outside Appetite"  # hard-rule veto always wins
    return decision


def verify_appetite_formula(df):
    """Uses the file's own STORED hard-rule flag (not our reconstruction) to measure the
    fit-score formula's accuracy in isolation - this is what gave R^2=0.977 originally.
    compute_appetite_fit() below has to use our own reconstructed flag instead, since a
    brand-new submission won't have the file's stored value - see module docstring for the
    known limitation that introduces."""
    from sklearn.linear_model import LinearRegression
    broker_tier_num = df["broker_tier"].map(BROKER_TIER_MAP).fillna(2)
    X = pd.DataFrame({
        "appetite_hard_rule_pass_flag": df["appetite_hard_rule_pass_flag"].astype(float),
        "target_growth_segment_flag": df["target_growth_segment_flag"].fillna(0),
        "financial_stability_score": df["financial_stability_score"].fillna(0),
        "broker_tier_num": broker_tier_num,
        "submission_completeness_pct": df["submission_completeness_pct"].fillna(0),
        "coverage_limit_requested_usd": df["coverage_limit_requested_usd"].fillna(0),
    })
    y = df["appetite_fit_score_0_100"]
    reg = LinearRegression().fit(X, y)
    pred = reg.predict(X)
    r2 = reg.score(X, y)
    mae = np.abs(pred - y).mean()
    coefs = {"intercept": reg.intercept_}
    coefs.update(dict(zip(X.columns, reg.coef_)))
    return {"r2": r2, "mae": mae, "coefficients": coefs}


if __name__ == "__main__":
    from data_loader import load_data
    df, schema, usable_for, output_cols = load_data()

    fit = verify_appetite_formula(df)
    print(f"Appetite Fit regression check: R^2={fit['r2']:.4f}, avg error={fit['mae']:.2f} points")
    print("Coefficients:", {k: round(v, 4) for k, v in fit["coefficients"].items()})

    computed_pass = hard_rule_pass(df)
    mismatch = (computed_pass != df["appetite_hard_rule_pass_flag"]).sum()
    print(f"\nHard-rule veto reconstruction: {mismatch} mismatches out of {len(df)} (should be 0)")

    AFC = fit["coefficients"]
    computed_fit = compute_appetite_fit(df, AFC)
    stored_fit = df["appetite_fit_score_0_100"]
    diff = pd.Series(computed_fit - stored_fit).abs()
    print(f"Appetite Fit reconstruction: avg diff = {diff.mean():.2f} points")

    computed_final = compute_appetite_capacity_score(df, AFC)
    stored_final = df["appetite_capacity_score_0_100"]
    diff2 = pd.Series(computed_final - stored_final).abs()
    print(f"Appetite/Capacity final score vs stored: avg diff = {diff2.mean():.2f} points, "
          f"correlation = {pd.Series(computed_final).corr(stored_final):.3f}")
