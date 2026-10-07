"""
Feature Map - data layer for the Streamlit "Feature Map" tab. Answers one question: which
real input fields feed Loss Propensity, Bind Propensity, and Appetite/Capacity, and where do
they overlap?

Every field list here is IMPORTED from the real scoring modules, never retyped - PROPERTY_FEATURES/
LIABILITY_FEATURES (lob_config.py), BIND_SOURCE_FIELDS (bind_propensity.py), APPETITE_FEATURES
(appetite_capacity.py). If a feature set in the actual scoring code changes, this file's output
changes with it automatically - there is nothing here to remember to update by hand.

Loss Propensity's circle is the UNION of the property and liability feature lists, since this
tab shows one project-wide picture rather than splitting by line of business (explicit scope
decision, 2026-10). The real nuance that gets lost by unioning - a handful of fields (TIV, the
3 CAT zone tiers, industry_class_naics) are only Loss-relevant for ONE of the two groups, even
though they're shown as part of the single "Loss Propensity" circle here - is preserved as a
per-field caveat in FIELD_NOTES rather than silently dropped, so the diagram never implies
something that isn't true for every line.
"""
from lob_config import PROPERTY_FEATURES, LIABILITY_FEATURES, ALL_LOBS, group_for_lob
from bind_propensity import BIND_SOURCE_FIELDS
from appetite_capacity import APPETITE_FEATURES

LOSS_FEATURES = sorted(set(PROPERTY_FEATURES) | set(LIABILITY_FEATURES))
BIND_FEATURES = sorted(set(BIND_SOURCE_FIELDS))
APPETITE_FEATURES_SET = sorted(set(APPETITE_FEATURES))

# Line-of-business filter for the chart - "All Lines" keeps the project-wide union; picking
# a specific line narrows Loss Propensity down to just that line's own real feature set
# (property lines and liability lines genuinely use different fields - see lob_config.py).
# Bind and Appetite never vary by line, so only Loss's circle actually changes when filtered.
LOB_OPTIONS = ["All Lines"] + ALL_LOBS


def loss_features_for(lob_selection):
    """The real Loss Propensity feature list for a given filter selection."""
    if lob_selection == "All Lines":
        return LOSS_FEATURES
    group = group_for_lob(lob_selection)
    return sorted(PROPERTY_FEATURES) if group == "property" else sorted(LIABILITY_FEATURES)


def compute_zones(loss_features):
    """The 7 Venn zones for a given Loss Propensity feature list - every field appears in
    exactly one. Bind and Appetite's own field sets never change; only which zone a field
    lands in shifts when Loss's feature list narrows to one line-of-business group."""
    _L, _B, _A = set(loss_features), set(BIND_FEATURES), set(APPETITE_FEATURES_SET)
    return {
        "loss":          sorted(_L - _B - _A),
        "bind":          sorted(_B - _L - _A),
        "appetite":      sorted(_A - _L - _B),
        "loss_bind":     sorted((_L & _B) - _A),
        "loss_appetite": sorted((_L & _A) - _B),
        "bind_appetite": sorted((_B & _A) - _L),
        "all3":          sorted(_L & _B & _A),
    }


# Default / "All Lines" view - also what the module's own __main__ self-check runs against.
ZONES = compute_zones(LOSS_FEATURES)

ZONE_LABELS = {
    "loss": "Loss Propensity only",
    "bind": "Bind Propensity only",
    "appetite": "Appetite / Capacity only",
    "loss_bind": "Loss Propensity + Bind Propensity",
    "loss_appetite": "Loss Propensity + Appetite / Capacity",
    "bind_appetite": "Bind Propensity + Appetite / Capacity",
    "all3": "All three scores",
}

FIELD_LABELS = {
    # property / physical
    "building_age_years": "Building Age", "built_to_current_code_flag": "Built to Current Code",
    "construction_type": "Construction Type", "num_stories": "Number of Stories",
    "square_footage": "Square Footage", "roof_age_years": "Roof Age", "roof_type": "Roof Type",
    "sprinklered_flag": "Sprinkler Status", "fire_protection_class_iso_ppc": "Fire Protection Class (ISO PPC)",
    "distance_to_fire_station_miles": "Distance to Fire Station", "distance_to_fire_hydrant_miles": "Distance to Fire Hydrant",
    "fire_alarm_monitoring_type": "Fire Alarm Monitoring", "occupancy_type": "Occupancy Type",
    # CAT / location
    "fema_flood_zone_code": "FEMA Flood Zone Code", "flood_zone_tier": "Flood Zone Tier",
    "earthquake_zone_tier": "Earthquake Zone Tier", "wildfire_zone_tier": "Wildfire Zone Tier",
    "wind_zone_tier": "Wind Zone Tier", "soil_liquefaction_class": "Soil Liquefaction Class",
    "primary_cat_peril_exposure": "Primary CAT Peril Exposure",
    "distance_to_coast_miles": "Distance to Coast", "distance_to_fault_line_miles": "Distance to Fault Line",
    "distance_to_nearest_water_body_miles": "Distance to Nearest Water Body",
    "total_insured_value_usd": "Total Insured Value",
    # workforce / liability
    "industry_class_naics": "Industry Class (NAICS)", "number_of_employees": "Number of Employees",
    "avg_wage_per_employee_usd": "Avg. Wage per Employee", "avg_employee_age": "Avg. Employee Age",
    "avg_weekly_work_hours": "Avg. Weekly Work Hours",
    # claims / financial history
    "claim_count_last_3_years": "Claim Count (3yr)", "large_loss_count_last_5_years": "Large-Loss Count (5yr)",
    "claims_free_years": "Claims-Free Years", "financial_stability_score": "Financial Stability Score",
    "safety_program_flag": "Safety Program in Place", "safety_inspection_findings": "Safety Inspection Findings",
    "business_tenure_years": "Business Tenure", "revenue_growth_trend_pct": "Revenue Growth Trend",
    "debt_to_equity_ratio": "Debt-to-Equity Ratio",
    # bind / broker / deal
    "broker_tier": "Broker Tier", "competing_quotes_count": "Competing Quotes Count",
    "coverage_limit_requested_usd": "Coverage Limit Requested", "risk_state": "State",
    "channel": "Channel",
    # appetite / compliance
    "sanctions_ofac_watchlist_flag": "Sanctions / OFAC Watchlist", "prior_insolvency_bankruptcy_flag": "Prior Insolvency / Bankruptcy",
    "restricted_operations_flag": "Restricted Operations", "region_lob_restricted_flag": "Region x LOB Restricted",
    "submission_completeness_pct": "Submission Completeness", "target_growth_segment_flag": "Target Growth Segment",
}

# Short, honest, one-line reason each field sits where it does. Fields that are only
# Loss-relevant for ONE of the two LOB groups (because this circle is a union across both -
# see module docstring) carry that caveat explicitly rather than implying universality.
FIELD_NOTES = {
    "building_age_years": "Older structures carry more physical risk - property lines only.",
    "built_to_current_code_flag": "Code-built construction resists loss better - property lines only.",
    "construction_type": "Fire-resistive vs. frame construction changes loss severity - property lines only.",
    "num_stories": "More stories, more at-risk floor area - property lines only.",
    "square_footage": "Larger footprint, larger loss exposure - property lines only.",
    "roof_age_years": "Older roofs fail more often - property lines only.",
    "roof_type": "Some roof types are more wind/fire resistant - property lines only.",
    "sprinklered_flag": "Sprinklers materially cut fire loss severity - property lines only.",
    "fire_protection_class_iso_ppc": "Standard ISO measure of local fire protection quality - property lines only.",
    "distance_to_fire_station_miles": "Farther from help, slower response, worse fire loss - property lines only.",
    "distance_to_fire_hydrant_miles": "Same logic as fire station distance - property lines only.",
    "fire_alarm_monitoring_type": "Monitored alarms catch fires earlier - property lines only.",
    "occupancy_type": "What the building is used for changes its risk profile - property lines only.",
    "fema_flood_zone_code": "Official flood-risk designation - property lines only.",
    "flood_zone_tier": "Flood risk tier - Loss-relevant for property lines only, but still triggers Appetite's CAT/TIV penalty for every line of business.",
    "earthquake_zone_tier": "Earthquake risk tier - Loss-relevant for property lines only, but still triggers Appetite's CAT/TIV penalty for every line of business.",
    "wildfire_zone_tier": "Wildfire risk tier - Loss-relevant for property lines only, but still triggers Appetite's CAT/TIV penalty for every line of business.",
    "wind_zone_tier": "Wind/hail risk tier - property lines only.",
    "soil_liquefaction_class": "Soil behavior in an earthquake - property lines only.",
    "primary_cat_peril_exposure": "The single dominant catastrophe peril type - property lines only; the one field added this project after a direct chi-square significance test (p=0.008) against real large claims.",
    "distance_to_coast_miles": "Proxy for hurricane/storm-surge exposure - property lines only.",
    "distance_to_fault_line_miles": "Proxy for earthquake exposure - property lines only.",
    "distance_to_nearest_water_body_miles": "Proxy for flood exposure - property lines only.",
    "total_insured_value_usd": "Bigger insured value, bigger potential loss (Loss, property lines only) - also a (near-zero-weight) Bind ingredient, and triggers Appetite's CAT/TIV penalty above $2M, for every line of business.",
    "industry_class_naics": "Different industries carry different claim patterns (Loss, liability lines only) - also drives Appetite's 15% industry-concentration cap, for every line of business.",
    "number_of_employees": "More people, more exposure to a workplace claim - liability lines only.",
    "avg_wage_per_employee_usd": "Proxies for job type/skill level and injury severity - liability lines only.",
    "avg_employee_age": "Older-worker claims often run longer and cost more - liability lines only.",
    "avg_weekly_work_hours": "More hours worked, more exposure - liability lines only, Workers Comp-relevant.",
    "claim_count_last_3_years": "Recent claim history is the single strongest predictor of future claims.",
    "large_loss_count_last_5_years": "A history of large losses raises the odds of another one.",
    "claims_free_years": "A clean recent record lowers expected risk.",
    "financial_stability_score": "Financially stressed businesses both claim more (Loss) and are a weaker strategic fit (Appetite).",
    "safety_program_flag": "An active safety program measurably lowers claim frequency.",
    "safety_inspection_findings": "Open safety deficiencies predict future claims.",
    "business_tenure_years": "Newer businesses have less of a track record to go on.",
    "revenue_growth_trend_pct": "Sharp revenue decline is a financial-distress signal.",
    "debt_to_equity_ratio": "Tested directly against real claims this project - too weak/confounded a signal to trust, dropped from the trained model on both groups, but still a declared candidate feature.",
    "broker_tier": "Drives 7 of Bind Propensity's real formula ingredients via real per-tier averages, and is a direct Appetite fit-score input.",
    "competing_quotes_count": "More competing quotes, lower odds of winning the account - Bind only.",
    "coverage_limit_requested_usd": "A Bind ingredient (near-zero fitted weight) and a (small) Appetite fit input.",
    "risk_state": "Drives Region for Bind's regional coefficients, and Appetite's real per-state reinsurance-headroom lookup.",
    "channel": "How the submission arrived (Broker/Direct/MGA) directly changes bind likelihood - Bind only.",
    "sanctions_ofac_watchlist_flag": "One of 4 hard-rule flags - any one true is an automatic decline, verified with zero exceptions.",
    "prior_insolvency_bankruptcy_flag": "Hard-rule flag - same as above.",
    "restricted_operations_flag": "Hard-rule flag - same as above.",
    "region_lob_restricted_flag": "Hard-rule flag - same as above.",
    "submission_completeness_pct": "A Bind ingredient and a (small) Appetite fit input - barely varies across the real book either way.",
    "target_growth_segment_flag": "A real, meaningful positive weight in the Appetite fit score for strategically targeted segments.",
}


# Compact labels for text actually placed ON the chart (the 32-field Loss-only region has no
# room for the full descriptive labels above) - the full label + the real rationale are still
# available on hover. Only fields that need shortening are listed; everything else falls back
# to FIELD_LABELS, which is already short enough to fit where it's shown.
SHORT_LABELS = {
    "building_age_years": "Building Age", "built_to_current_code_flag": "Built to Code",
    "business_tenure_years": "Business Tenure", "claim_count_last_3_years": "Claims (3yr)",
    "claims_free_years": "Claims-Free Yrs", "construction_type": "Construction",
    "debt_to_equity_ratio": "Debt/Equity", "distance_to_coast_miles": "Dist. Coast",
    "distance_to_fault_line_miles": "Dist. Fault Line", "distance_to_fire_hydrant_miles": "Dist. Hydrant",
    "distance_to_fire_station_miles": "Dist. Fire Station", "distance_to_nearest_water_body_miles": "Dist. Water Body",
    "fema_flood_zone_code": "FEMA Flood Code", "fire_alarm_monitoring_type": "Fire Alarm",
    "fire_protection_class_iso_ppc": "Fire Protect. Class", "large_loss_count_last_5_years": "Large Loss (5yr)",
    "num_stories": "Stories", "number_of_employees": "Employees", "occupancy_type": "Occupancy",
    "primary_cat_peril_exposure": "Primary CAT Peril", "revenue_growth_trend_pct": "Revenue Trend",
    "roof_age_years": "Roof Age", "roof_type": "Roof Type", "safety_inspection_findings": "Safety Findings",
    "safety_program_flag": "Safety Program", "soil_liquefaction_class": "Soil Liquefaction",
    "sprinklered_flag": "Sprinklered", "square_footage": "Sq. Footage", "wind_zone_tier": "Wind Zone",
    "avg_employee_age": "Avg. Employee Age", "avg_wage_per_employee_usd": "Avg. Wage",
    "avg_weekly_work_hours": "Weekly Hours",
}


def field_display(field_id):
    """(label, note) for a field, falling back gracefully if either dict is ever missing an entry."""
    label = FIELD_LABELS.get(field_id, field_id.replace("_", " ").title())
    note = FIELD_NOTES.get(field_id, "")
    return label, note


def zone_counts():
    return {z: len(fields) for z, fields in ZONES.items()}


# Real 3-circle geometry, sized roughly to each score's field count (Loss has by far the
# most fields and needs by far the most room) and positioned so every pairwise overlap AND
# the triple-overlap center are real, non-empty regions - verified algebraically (each
# pairwise center-distance sits at ~60% of that pair's summed radii, a safe, generous overlap
# for any combination of unequal radii) before a single pixel was drawn, since there's no way
# to visually proof this after the fact from here.
_CIRCLES = {"loss": (270, 260, 245), "bind": (520, 260, 170), "appetite": (440, 465, 200)}

# Each zone's on-chart text box (x, y, width, height) in the same SVG coordinate space as
# _CIRCLES above - hand-placed at each zone's real geometric region (solo zones pushed toward
# that circle's own exclusive area, overlaps at the midpoint between the relevant circles),
# then checked against every circle's actual center+radius so a box only ever lands inside
# the circle(s) that zone belongs to. Loss's box is deliberately the biggest, by far, to hold
# its 32 fields.
_ZONE_BOXES = {
    "loss":          (50, 90, 230, 230),
    "bind":          (525, 180, 120, 65),
    "appetite":      (370, 510, 170, 95),
    "loss_bind":     (350, 222, 80, 36),
    "loss_appetite": (285, 325, 140, 78),
    "bind_appetite": (455, 345, 125, 68),
    "all3":          (368, 306, 85, 42),
}


# Design width the chart is laid out for - the SVG never renders wider than this, which
# makes its rendered height fully predictable (aspect ratio is fixed by the viewBox) instead
# of depending on whatever width the browser happens to give it. That predictability is what
# lets the Streamlit component height be set to match exactly, with no scrollbar and nothing
# clipped.
_DESIGN_WIDTH = 980
_VIEWBOX_W, _VIEWBOX_H = 720, 700
COMPONENT_HEIGHT = int(_DESIGN_WIDTH * _VIEWBOX_H / _VIEWBOX_W) + 50  # +50: hint line + padding


def build_venn_component(is_dark, zones=None):
    """Self-contained HTML/SVG - a real, proportionally-sized 3-circle Venn with every field
    written directly inside its own region (via SVG <foreignObject>, for real HTML text
    wrapping), nothing below the chart, nothing behind a click. `zones` defaults to the
    project-wide "All Lines" view (module-level ZONES) - pass the output of compute_zones()
    for a line-of-business-filtered view instead; the circle geometry and box positions stay
    fixed either way, only which fields land in which box (and whether a box has anything in
    it at all) changes."""
    zones = zones if zones is not None else ZONES
    ink = "#E7ECF3" if is_dark else "#16202C"
    muted = "#A9B4C2" if is_dark else "#4A5568"
    page_bg = "#0F1622" if is_dark else "#F7F9FC"
    colors = {
        "loss": "#7DA6F5" if is_dark else "#2E5090",
        "bind": "#3FD9C7" if is_dark else "#0E8A7D",
        "appetite": "#C9A3F2" if is_dark else "#7B4FA6",
    }
    chip_ink = "#0B1017" if is_dark else "#16202C"

    def field_span(field_id, compact=False):
        label = (SHORT_LABELS.get(field_id) or FIELD_LABELS.get(field_id, field_id)) if compact \
            else FIELD_LABELS.get(field_id, field_id)
        note = FIELD_NOTES.get(field_id, "").replace('"', "&quot;")
        return f'<span class="vn-f" title="{note}">{label}</span>'

    def zone_box(zone_id, dots, columns=1, compact=False):
        fields = zones[zone_id]
        if not fields:
            return ""  # an empty zone draws nothing at all - no awkward "none" placeholder
        x, y, w, h = _ZONE_BOXES[zone_id]
        dot_row = "".join(f'<i class="vn-dot" style="background:{colors[c]}"></i>' for c in dots) \
            if len(dots) > 1 else ""  # only overlap zones get a marker - solo zones are unambiguous by position/color
        body = f'<div class="vn-flist" style="column-count:{columns};">' \
               + "".join(field_span(f, compact) for f in fields) + "</div>"
        head = f'<div class="vn-dots">{dot_row}</div>' if dot_row else ""
        return f"""<foreignObject x="{x}" y="{y}" width="{w}" height="{h}">
          <div xmlns="http://www.w3.org/1999/xhtml" class="vn-box">{head}{body}</div>
        </foreignObject>"""

    boxes = "".join([
        zone_box("loss", ["loss"], columns=2, compact=True),
        zone_box("bind", ["bind"], columns=1),
        zone_box("appetite", ["appetite"], columns=1),
        zone_box("loss_bind", ["loss", "bind"], columns=1),
        zone_box("loss_appetite", ["loss", "appetite"], columns=1),
        zone_box("bind_appetite", ["bind", "appetite"], columns=1),
        zone_box("all3", ["loss", "bind", "appetite"], columns=1),
    ])

    L, B, A = _CIRCLES["loss"], _CIRCLES["bind"], _CIRCLES["appetite"]

    return f"""
<div style="font-family:'Inter',-apple-system,sans-serif; background:{page_bg}; padding:4px;">
  <style>
    .vn-hint {{ color:{muted}; font-size:0.85rem; margin:0 0 8px 2px; }}
    .vn-chart-wrap {{ max-width:{_DESIGN_WIDTH}px; margin:0 auto; }}
    svg {{ width:100%; height:auto; display:block; }}
    .vn-title {{ font-size:15px; font-weight:800; }}
    .vn-box {{
      font-family:'Inter',-apple-system,sans-serif; width:100%; height:100%; overflow:hidden;
      box-sizing:border-box; padding:2px 5px;
    }}
    .vn-dots {{ display:flex; gap:3px; margin-bottom:3px; }}
    .vn-dot {{ width:7px; height:7px; border-radius:50%; display:inline-block; }}
    .vn-flist {{ column-gap:8px; }}
    .vn-f {{
      display:block; font-size:9.3px; line-height:1.35; color:{chip_ink}; cursor:help;
      break-inside:avoid; padding:0.5px 0;
    }}
  </style>
  <div class="vn-hint">Every field each score uses, placed directly where it lives. Hover any field for why it's there.</div>
  <div class="vn-chart-wrap">
    <svg viewBox="0 0 {_VIEWBOX_W} {_VIEWBOX_H}" xmlns="http://www.w3.org/2000/svg">
      <circle cx="{L[0]}" cy="{L[1]}" r="{L[2]}" fill="{colors['loss']}" fill-opacity="0.26" stroke="{colors['loss']}" stroke-width="2.5"/>
      <circle cx="{B[0]}" cy="{B[1]}" r="{B[2]}" fill="{colors['bind']}" fill-opacity="0.26" stroke="{colors['bind']}" stroke-width="2.5"/>
      <circle cx="{A[0]}" cy="{A[1]}" r="{A[2]}" fill="{colors['appetite']}" fill-opacity="0.26" stroke="{colors['appetite']}" stroke-width="2.5"/>
      <text class="vn-title" x="{L[0]-L[2]+20}" y="{L[1]-L[2]+34}" fill="{colors['loss']}">Loss Propensity</text>
      <text class="vn-title" x="{B[0]+B[2]-18}" y="{B[1]-B[2]+34}" fill="{colors['bind']}" text-anchor="end">Bind Propensity</text>
      <text class="vn-title" x="{A[0]}" y="{A[1]+A[2]-16}" fill="{colors['appetite']}" text-anchor="middle">Appetite / Capacity</text>
      {boxes}
    </svg>
  </div>
</div>
"""



if __name__ == "__main__":
    print(f"Loss Propensity : {len(LOSS_FEATURES)} fields")
    print(f"Bind Propensity : {len(BIND_FEATURES)} fields")
    print(f"Appetite        : {len(APPETITE_FEATURES_SET)} fields")
    print(f"Total unique    : {len(_LOSS | _BIND | _APP)} fields")
    print()
    for z, fields in ZONES.items():
        print(f"{ZONE_LABELS[z]:42s} ({len(fields)}): {fields}")
    missing_labels = [f for f in (_LOSS | _BIND | _APP) if f not in FIELD_LABELS]
    missing_notes = [f for f in (_LOSS | _BIND | _APP) if f not in FIELD_NOTES]
    print(f"\nFields missing a label: {missing_labels}")
    print(f"Fields missing a note : {missing_notes}")
