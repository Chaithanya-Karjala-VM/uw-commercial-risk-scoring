"""
Fixed reference text for the Underwriting Decision Matrix, matching the real Excel
Dashboard sheet content exactly (verified against the source file's own formulas).
"""

RISK_TIER_ORDER = ["Preferred", "Standard", "High", "Decline"]

# machine-usable version of the bands above (text-only until now, e.g. "55 - 71") - single
# source of truth for the numeric cutoffs, so scoring.py, the dashboard, and any future caller
# all classify a uw_score_v2 into a tier the exact same way, instead of each hardcoding 72/55/38
RISK_TIER_CUTOFFS = [(72, "Preferred"), (55, "Standard"), (38, "High"), (0, "Decline")]


def tier_for_score(uw_score):
    for cutoff, tier in RISK_TIER_CUTOFFS:
        if uw_score >= cutoff:
            return tier
    return "Decline"


RISK_TIER_INFO = {
    "Preferred": {
        "uw_score_band": "≥ 72",
        "what_it_means": "Good risk, within appetite, likely to bind",
        "uw_action": "Fast-track — no rate action",
        "pricing_guidance": "No adjustment",
    },
    "Standard": {
        "uw_score_band": "55 – 71",
        "what_it_means": "Solid risk, minor watch items",
        "uw_action": "Rate at technical price",
        "pricing_guidance": "Technical price",
    },
    "High": {
        "uw_score_band": "38 – 54",
        "what_it_means": "Marginal on one or more of the 3 scores",
        "uw_action": "Rate adjustment + deductible review",
        "pricing_guidance": "+10-20% rate; increase deductible",
    },
    "Decline": {
        "uw_score_band": "< 38, or any hard-rule fail",
        "what_it_means": "Poor loss view, outside appetite, or capacity-constrained",
        "uw_action": "Refer to senior underwriter",
        "pricing_guidance": "Refer — do not auto-quote",
    },
}
