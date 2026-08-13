"""
Which Lines of Business are property risks vs. liability/operations risks, and which inputs
are genuinely relevant to each - built from checking the actual data (see conversation:
property-peril fields were being applied uniformly to all 8 lines, including ones where they
have no real bearing, while genuinely relevant liability fields like industry class and
employee count existed in the data but were never used as model inputs at all).
"""

PROPERTY_LOBS = ["Commercial Property", "Business Owners Package (BOP)", "Inland Marine"]
LIABILITY_LOBS = ["Workers Compensation", "General Liability", "Professional Liability",
                   "Umbrella/Excess", "Commercial Auto"]

# Fields that describe a BUILDING - relevant only where the exposure is a physical structure.
PROPERTY_ONLY_FEATURES = [
    "year_built", "building_age_years", "construction_type", "square_footage",
    "roof_age_years", "sprinklered_flag", "fire_protection_class_iso_ppc",
    "fema_flood_zone_code", "flood_zone_tier", "distance_to_nearest_water_body_miles",
    "distance_to_coast_miles", "earthquake_zone_tier", "distance_to_fault_line_miles",
    "soil_liquefaction_class", "wildfire_zone_tier", "wind_zone_tier",
    "total_insured_value_usd",
]

# Fields that describe the BUSINESS/OPERATIONS - relevant to liability, WC, auto, etc.,
# where the risk comes from what the business does and who it employs, not a building.
LIABILITY_ONLY_FEATURES = [
    "industry_class_naics", "number_of_employees", "avg_wage_per_employee_usd",
    "avg_employee_age",
]

# Relevant to every line regardless of type.
COMMON_FEATURES = [
    "prior_claim_count", "prior_total_incurred_usd", "financial_stability_score",
    "safety_program_flag", "safety_inspection_findings",
]

PROPERTY_FEATURES = PROPERTY_ONLY_FEATURES + COMMON_FEATURES
LIABILITY_FEATURES = LIABILITY_ONLY_FEATURES + COMMON_FEATURES

# Empirical premium rate per LOB, computed from the real book on the correct exposure basis
# (TIV for property, payroll for liability) - replaces asking a user to type in a premium
# blindly, which was swamping every other input in the score (see conversation).
PREMIUM_RATE_BY_LOB = {
    "Commercial Property": 0.00499,
    "Business Owners Package (BOP)": 0.00545,
    "Inland Marine": 0.00534,
    "Workers Compensation": 0.00737,
    "General Liability": 0.00226,
    "Professional Liability": 0.00187,
    "Umbrella/Excess": 0.00226,
    "Commercial Auto": 0.00175,
}


def group_for_lob(lob):
    return "property" if lob in PROPERTY_LOBS else "liability"


def features_for_lob(lob):
    return PROPERTY_FEATURES if lob in PROPERTY_LOBS else LIABILITY_FEATURES


def compute_premium(lob, total_insured_value_usd=None, number_of_employees=None,
                     avg_wage_per_employee_usd=None):
    rate = PREMIUM_RATE_BY_LOB[lob]
    if lob in PROPERTY_LOBS:
        return round(total_insured_value_usd * rate, -1)
    else:
        payroll = (number_of_employees or 0) * (avg_wage_per_employee_usd or 0)
        return round(payroll * rate, -1)
