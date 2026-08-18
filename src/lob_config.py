"""
Property vs. liability line grouping and feature sets, rebuilt for the new
UW_Risk_Score_Commercial_F.xlsx schema.

Everything below reflects lessons already verified against this specific file:
  - claims_free_years is confirmed NOT leakage in this version (66% of rows with a current
    claim show a positive value - genuinely backward-looking now) and is included.
  - claim_count_last_3_years and large_loss_count_last_5_years are the explicit-window
    fields added specifically in response to manager feedback - included, and preferred
    over the vaguer prior_claim_count/prior_total_incurred_usd pair.
  - Several fields tagged "General" rather than "Loss" in the dictionary are genuinely
    risk-relevant and were checked before including: roof_type, num_stories,
    built_to_current_code_flag, distance_to_fire_station/hydrant, business_tenure_years,
    revenue_growth_trend_pct, debt_to_equity_ratio, avg_weekly_work_hours (WC-specific).
"""

PROPERTY_LOBS = ["Commercial Property", "Business Owners Package (BOP)", "Inland Marine"]
LIABILITY_LOBS = ["Workers Compensation", "General Liability", "Professional Liability",
                   "Umbrella/Excess", "Commercial Auto"]

PROPERTY_ONLY_FEATURES = [
    "year_built", "building_age_years", "built_to_current_code_flag", "construction_type",
    "num_stories", "square_footage", "roof_age_years", "roof_type", "sprinklered_flag",
    "fire_protection_class_iso_ppc", "distance_to_fire_station_miles",
    "distance_to_fire_hydrant_miles", "fire_alarm_monitoring_type",
    "fema_flood_zone_code", "flood_zone_tier", "distance_to_nearest_water_body_miles",
    "distance_to_coast_miles", "earthquake_zone_tier", "distance_to_fault_line_miles",
    "soil_liquefaction_class", "wildfire_zone_tier", "wind_zone_tier",
    "total_insured_value_usd", "occupancy_type",
]

LIABILITY_ONLY_FEATURES = [
    "industry_class_naics", "number_of_employees", "avg_wage_per_employee_usd",
    "avg_employee_age", "avg_weekly_work_hours",
]

# common to both - includes the new explicit-window claims history fields, preferred over
# the old vague prior_claim_count/prior_total_incurred_usd pair per manager feedback
COMMON_FEATURES = [
    "claim_count_last_3_years", "large_loss_count_last_5_years", "claims_free_years",
    "financial_stability_score", "safety_program_flag", "safety_inspection_findings",
    "business_tenure_years", "revenue_growth_trend_pct", "debt_to_equity_ratio",
]

PROPERTY_FEATURES = PROPERTY_ONLY_FEATURES + COMMON_FEATURES
LIABILITY_FEATURES = LIABILITY_ONLY_FEATURES + COMMON_FEATURES

# these are outcomes of a claim that's already happened in the CURRENT period, or are
# otherwise not knowable before the fact - excluded from every feature set regardless of tag
CLAIM_OUTCOME_COLS = [
    "claim_count_total", "claim_count_open", "claim_count_settled", "primary_cause_of_claim",
    "claim_nature", "claim_severity_avg_usd", "total_incurred_claim_amount_usd",
    "large_claim_flag", "large_claim_count", "catastrophic_loss_flag",
    "catastrophic_loss_amount_usd", "hospital_visit_count", "treatment_type",
    "litigated_claim_flag", "litigated_claim_count", "prior_claim_count",
    "prior_total_incurred_usd",  # superseded by the explicit-window fields above
]


def group_for_lob(lob):
    return "property" if lob in PROPERTY_LOBS else "liability"


def features_for_lob(lob):
    return PROPERTY_FEATURES if lob in PROPERTY_LOBS else LIABILITY_FEATURES


# These four fields are genuinely ordered (Low < Moderate < High < Severe) but one-hot
# encoding throws that ordering away, which is exactly why they came back non-monotonic
# when tested directly - Low flood scored WORSE than Moderate/High, backwards, the same
# root cause as the financial-stability bug found earlier but on a categorical field the
# numeric monotonic_cst mechanism never covered. Encoding them as ordinal integers instead
# lets the same monotonic-constraint mechanism apply here too.
ORDINAL_TIER_FIELDS = ["flood_zone_tier", "earthquake_zone_tier", "wildfire_zone_tier", "wind_zone_tier"]
ORDINAL_TIER_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}


def encode_ordinal_tiers(df):
    df = df.copy()
    for c in ORDINAL_TIER_FIELDS:
        if c in df.columns:
            df[c] = df[c].map(ORDINAL_TIER_MAP)
    return df
