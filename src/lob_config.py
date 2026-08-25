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
from datetime import date

PROPERTY_LOBS = ["Commercial Property", "Business Owners Package (BOP)", "Inland Marine"]
LIABILITY_LOBS = ["Workers Compensation", "General Liability", "Professional Liability",
                   "Umbrella/Excess", "Commercial Auto"]
ALL_LOBS = PROPERTY_LOBS + LIABILITY_LOBS  # closed set the models were trained on - anything
                                            # outside this list is genuinely new to the system

PROPERTY_ONLY_FEATURES = [
    # year_built is deliberately NOT included as a model feature (it's still a live form
    # input) - it's deterministically redundant with building_age_years (correlation
    # -0.9999999999999999 in the real book, confirmed 2026-08), and having both fed to the
    # model as separate features was pure multicollinearity causing exactly the kind of erratic,
    # non-monotonic behavior found across an 8-LOB audit. building_age_years alone (already
    # monotonically constrained) carries the same information cleanly.
    "building_age_years", "built_to_current_code_flag", "construction_type",
    "num_stories", "square_footage", "roof_age_years", "roof_type", "sprinklered_flag",
    "fire_protection_class_iso_ppc", "distance_to_fire_station_miles",
    "distance_to_fire_hydrant_miles", "fire_alarm_monitoring_type",
    "fema_flood_zone_code", "flood_zone_tier", "distance_to_nearest_water_body_miles",
    "distance_to_coast_miles", "earthquake_zone_tier", "distance_to_fault_line_miles",
    "soil_liquefaction_class", "wildfire_zone_tier", "wind_zone_tier",
    "total_insured_value_usd", "occupancy_type",
    # added 2026-08: chi-square tested against large_claim_flag specifically (property),
    # p=0.008 - genuinely significant, unlike every other untested categorical field checked
    # (business_description, occupancy_type-alone, industry, water-body-type, region all came
    # back p>0.05 - noise from thin per-category counts on a rare target). This one carries
    # real, distinct signal from the existing zone-tier fields: it names the peril TYPE
    # directly (Wildfire / Flood-Storm Surge / Earthquake / Wind-Hail / Elevated-Multiple
    # Perils / Low-Minimal) rather than a severity tier per peril, which the tier fields
    # already cover from a different angle.
    "primary_cat_peril_exposure",
]
PRIMARY_CAT_PERIL_OPTIONS = ["Low/Minimal CAT Exposure", "Flood/Storm Surge", "Wildfire",
                             "Earthquake", "Wind/Hail", "Elevated — Multiple Perils"]

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


# Single source for "today" across the whole app - was previously hand-typed as the literal
# number 2026 in three separate places (Submission form's max Year Built, the building-age
# calculation, and a test boundary value). That rots the instant the calendar turns over;
# importing this everywhere means the app is never wrong about what year it is.
CURRENT_YEAR = date.today().year
