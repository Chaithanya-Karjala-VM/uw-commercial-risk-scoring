"""
Property vs. liability line grouping and feature sets.

Notes on feature selection:
  - claims_free_years is genuinely backward-looking in this schema (verified: not derived
    from the current-period outcome) and is safe to use.
  - claim_count_last_3_years / large_loss_count_last_5_years are explicit-window fields,
    used in place of the ambiguous prior_claim_count / prior_total_incurred_usd pair.
  - A few fields tagged "General" rather than "Loss" in the dictionary are genuinely
    risk-relevant and included anyway: roof_type, num_stories, built_to_current_code_flag,
    distance_to_fire_station/hydrant, business_tenure_years, revenue_growth_trend_pct,
    debt_to_equity_ratio, avg_weekly_work_hours (WC-specific).
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

# common to both lines - explicit-window claims history preferred over the ambiguous
# prior_claim_count / prior_total_incurred_usd pair
COMMON_FEATURES = [
    "claim_count_last_3_years", "large_loss_count_last_5_years", "claims_free_years",
    "financial_stability_score", "safety_program_flag", "safety_inspection_findings",
    "business_tenure_years", "revenue_growth_trend_pct", "debt_to_equity_ratio",
]

PROPERTY_FEATURES = PROPERTY_ONLY_FEATURES + COMMON_FEATURES
LIABILITY_FEATURES = LIABILITY_ONLY_FEATURES + COMMON_FEATURES

# these columns only exist once a claim has already happened - excluded from every
# feature set as potential leakage, regardless of their dictionary tag
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


# Low/Moderate/High/Severe are ordered, but one-hot encoding discards that order, which
# breaks monotonic constraints for these fields specifically. Encoding as ordinal integers
# instead lets monotonic_cst apply here the same way it does for numeric fields.
ORDINAL_TIER_FIELDS = ["flood_zone_tier", "earthquake_zone_tier", "wildfire_zone_tier", "wind_zone_tier"]
ORDINAL_TIER_MAP = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}


def encode_ordinal_tiers(df):
    df = df.copy()
    for c in ORDINAL_TIER_FIELDS:
        if c in df.columns:
            df[c] = df[c].map(ORDINAL_TIER_MAP)
    return df


# Derived directly from the data - confirmed a clean 1:1 mapping, every state maps to
# exactly one region with zero exceptions across all 20,000 rows. Lets the Submission form
# collect State only, with Region derived automatically for whichever model needs it.
STATE_TO_REGION = {
    'AL': 'Gulf Coast', 'AZ': 'Southwest', 'CA': 'West', 'CO': 'Mountain', 'CT': 'Northeast',
    'FL': 'Southeast', 'GA': 'Southeast', 'ID': 'Mountain', 'IL': 'Midwest', 'IN': 'Midwest',
    'LA': 'Gulf Coast', 'MA': 'Northeast', 'MI': 'Midwest', 'MO': 'Midwest', 'MS': 'Gulf Coast',
    'MT': 'Mountain', 'NC': 'Southeast', 'NJ': 'Northeast', 'NM': 'Southwest', 'NV': 'West',
    'NY': 'Northeast', 'OH': 'Midwest', 'OK': 'Southwest', 'OR': 'Pacific Northwest',
    'PA': 'Northeast', 'SC': 'Southeast', 'TX': 'Gulf Coast', 'UT': 'Mountain',
    'WA': 'Pacific Northwest', 'WI': 'Midwest', 'WY': 'Mountain',
}


def region_for_state(state):
    return STATE_TO_REGION.get(state, "Midwest")  # fallback to the most common region
