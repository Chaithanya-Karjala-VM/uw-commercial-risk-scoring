"""
End-to-end model behavior test suite. Not part of the scoring pipeline itself - a
standalone check that everything still behaves correctly after any retrain or code change.
Run: python test_model_behavior.py
"""
import itertools
import pandas as pd
from scoring import load_models, score
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS

models = load_models()
ALL_LOBS = PROPERTY_LOBS + LIABILITY_LOBS
TIER_OPTIONS = ["Low", "Moderate", "High", "Severe"]
BROKER_TIERS = ["Platinum", "Gold", "Silver", "Bronze"]


def base_row(lob, level="mid"):
    is_prop = lob in PROPERTY_LOBS
    lvl = {"best": 0, "mid": 1, "worst": 2}[level]
    safety = ["No Issues Noted", "Minor Issues Noted", "Major Deficiencies Identified"][lvl]
    row = {
        "line_of_business": lob, "policy_term_months": 12,
        "claim_count_last_3_years": [0, 1, 5][lvl], "large_loss_count_last_5_years": [0, 0, 2][lvl],
        "claims_free_years": [10, 3, 0][lvl], "financial_stability_score": [90, 55, 15][lvl],
        "safety_program_flag": [True, False, False][lvl], "safety_inspection_findings": safety,
        "business_tenure_years": [20, 10, 1][lvl], "revenue_growth_trend_pct": [5, 2, -15][lvl],
        "debt_to_equity_ratio": [0.3, 1.0, 3.5][lvl], "broker_tier": BROKER_TIERS[lvl] if lvl < 3 else "Bronze",
        "broker_hit_bind_ratio_pct": [65, 40, 20][lvl], "broker_renewal_retention_pct": [97, 92, 85][lvl],
        "broker_product_diversity_count": [6, 3, 1][lvl], "broker_geo_diversity_count": [8, 3, 1][lvl],
        "broker_profitability_score_0_100": [70, 45, 25][lvl], "quote_to_bind_days": [10, 20, 45][lvl],
        "competing_quotes_count": [1, 2, 6][lvl], "submission_completeness_pct": [95, 85, 70][lvl],
        "sanctions_ofac_watchlist_flag": 0, "prior_insolvency_bankruptcy_flag": 0,
        "restricted_operations_flag": 0, "region_lob_restricted_flag": 0,
        "target_growth_segment_flag": 1 if lvl == 0 else 0,
        "risk_state": ["CA", "TX", "OK"][lvl], "coverage_limit_requested_usd": [500000, 500000, 1500000][lvl],
        "industry_class_naics": "Manufacturing", "number_of_employees": 25,
        "avg_wage_per_employee_usd": 55000, "avg_employee_age": 40, "avg_weekly_work_hours": 40.0,
        "construction_type": ["Fire Resistive", "Joisted Masonry", "Frame"][lvl],
        "year_built": [2023, 1995, 1930][lvl], "building_age_years": [3, 31, 96][lvl],
        "built_to_current_code_flag": [True, False, False][lvl], "num_stories": 2,
        "square_footage": 5000, "total_insured_value_usd": 500000, "occupancy_type": "Office",
        "roof_age_years": [2, 15, 40][lvl], "roof_type": ["Tile", "Shingle", "Flat/Built-Up"][lvl],
        "sprinklered_flag": ["Y", "Y", "N"][lvl], "fire_protection_class_iso_ppc": [1, 5, 10][lvl],
        "distance_to_fire_station_miles": [0.5, 5, 20][lvl], "distance_to_fire_hydrant_miles": [0.1, 0.5, 3.0][lvl],
        "fire_alarm_monitoring_type": ["Central Station Monitored", "Central Station Monitored", "Local Alarm Only"][lvl],
        "flood_zone_tier": ["Low", "Low", "Severe"][lvl], "fema_flood_zone_code": ["X", "X", "VE"][lvl],
        "earthquake_zone_tier": ["Low", "Low", "Severe"][lvl],
        "soil_liquefaction_class": ["A - Hard Rock", "B - Rock", "F - Special Study Required"][lvl],
        "wildfire_zone_tier": ["Low", "Low", "Severe"][lvl], "wind_zone_tier": ["Low", "Moderate", "Severe"][lvl],
        "distance_to_coast_miles": [400.0, 60.0, 1.0][lvl], "distance_to_fault_line_miles": [300.0, 60.0, 1.0][lvl],
        "distance_to_nearest_water_body_miles": [40.0, 3.0, 0.1][lvl],
    }
    return row


def score_one(row):
    return score(pd.DataFrame([row]), models).iloc[0]


def stage_1_best_worst():
    print("\n=== Stage 1: Best vs worst case, all 8 lines, all 4 scores ===")
    fails = 0
    for lob in ALL_LOBS:
        best, worst = score_one(base_row(lob, "best")), score_one(base_row(lob, "worst"))
        ok = (best["loss_propensity_score_v2"] < worst["loss_propensity_score_v2"]
              and best["bind_propensity_score_v2"] > worst["bind_propensity_score_v2"]
              and best["appetite_capacity_score_v2"] >= worst["appetite_capacity_score_v2"]
              and best["uw_score_v2"] > worst["uw_score_v2"])
        if not ok:
            fails += 1
            print(f"  FAIL: {lob}")
    print(f"  {len(ALL_LOBS) - fails}/{len(ALL_LOBS)} lines correct direction on all 4 scores")


def stage_2_monotonicity():
    print("\n=== Stage 2: Monotonicity sweeps, all inputs, all 8 lines ===")
    sweeps = {
        "financial_stability_score": [10, 30, 50, 70, 90, 99], "claim_count_last_3_years": [0, 2, 4, 6, 9],
        "large_loss_count_last_5_years": [0, 1, 2, 3], "claims_free_years": [0, 3, 5, 8, 10],
        "debt_to_equity_ratio": [0.0, 1.0, 2.5, 5.0], "broker_hit_bind_ratio_pct": [10, 30, 50, 70, 90],
        "competing_quotes_count": [0, 2, 4, 6, 10], "quote_to_bind_days": [5, 20, 40, 80, 120],
    }
    tier_sweeps = {"flood_zone_tier": TIER_OPTIONS, "earthquake_zone_tier": TIER_OPTIONS,
                   "wildfire_zone_tier": TIER_OPTIONS, "wind_zone_tier": TIER_OPTIONS}
    total, non_mono = 0, []
    for lob in ALL_LOBS:
        base = base_row(lob, "mid")
        for field, values in sweeps.items():
            total += 1
            scores_list = [score_one({**base, field: v})["loss_propensity_score_v2"] for v in values]
            if not (all(scores_list[i] <= scores_list[i+1] + 0.15 for i in range(len(scores_list)-1))
                    or all(scores_list[i] >= scores_list[i+1] - 0.15 for i in range(len(scores_list)-1))):
                non_mono.append((lob, field, [round(v, 1) for v in scores_list]))
        if lob in PROPERTY_LOBS:
            for field, values in tier_sweeps.items():
                total += 1
                scores_list = [score_one({**base, field: v})["loss_propensity_score_v2"] for v in values]
                if not all(scores_list[i] <= scores_list[i+1] + 0.15 for i in range(len(scores_list)-1)):
                    non_mono.append((lob, field, [round(v, 1) for v in scores_list]))
    print(f"  {total} checks run, {len(non_mono)} non-monotonic")
    for lob, field, vals in non_mono:
        print(f"    {lob} | {field} | {vals}")


def stage_3_bind_sensitivity():
    print("\n=== Stage 3: Bind Propensity sensitivity + independence from risk ===")
    base = base_row("Commercial Property", "mid")
    for field, values in {
        "broker_hit_bind_ratio_pct": [10, 30, 50, 70, 90], "broker_renewal_retention_pct": [50, 70, 85, 95],
        "competing_quotes_count": [0, 2, 4, 6, 10], "quote_to_bind_days": [5, 20, 40, 80, 120],
    }.items():
        vals = [round(score_one({**base, field: v})["bind_propensity_score_v2"], 1) for v in values]
        print(f"  {field:28s} swing={max(vals)-min(vals):5.1f}  {vals}")

    r1, r2 = base_row("Commercial Property", "best"), base_row("Commercial Property", "worst")
    for f in ["broker_tier", "broker_hit_bind_ratio_pct", "broker_renewal_retention_pct",
              "broker_product_diversity_count", "broker_geo_diversity_count",
              "broker_profitability_score_0_100", "quote_to_bind_days", "competing_quotes_count",
              "submission_completeness_pct"]:
        r2[f] = r1[f]
    s1, s2 = score_one(r1), score_one(r2)
    diff = abs(s1["bind_propensity_score_v2"] - s2["bind_propensity_score_v2"])
    print(f"  Independence check: Bind changed by {diff:.2f} pts when only risk factors varied (want ~0)")


def stage_4_appetite_gate():
    print("\n=== Stage 4: Appetite hard-rule gate - all 16 flag combinations ===")
    base = base_row("Commercial Property", "mid")
    flags = ["sanctions_ofac_watchlist_flag", "prior_insolvency_bankruptcy_flag",
             "restricted_operations_flag", "region_lob_restricted_flag"]
    violations = 0
    for combo in itertools.product([0, 1], repeat=4):
        r = {**base, **dict(zip(flags, combo))}
        s = score_one(r)
        if any(combo) and s["appetite_capacity_score_v2"] != 0:
            violations += 1
            print(f"  VIOLATION: flags={combo} -> Appetite={s['appetite_capacity_score_v2']}")
    print(f"  16/16 combinations tested, {violations} gate violations")


def stage_5_dropdowns():
    print("\n=== Stage 5: Every dropdown option, every relevant line ===")
    dropdowns = {
        "construction_type": ["Fire Resistive", "Non-Combustible", "Masonry Non-Combustible", "Joisted Masonry", "Frame"],
        "roof_type": ["Tile", "Metal", "Membrane/TPO", "Shingle", "Flat/Built-Up"],
        "fema_flood_zone_code": ["X", "X (shaded)", "A", "AE", "V", "VE"],
        "soil_liquefaction_class": ["A - Hard Rock", "B - Rock", "C - Dense Soil", "D - Stiff Soil", "E - Soft Soil", "F - Special Study Required"],
        "occupancy_type": ["Office", "Retail Store", "Warehouse", "Distribution Center", "Restaurant", "Mixed Use", "Manufacturing Plant", "Habitational/Apartments"],
        "fire_alarm_monitoring_type": ["Central Station Monitored", "Local Alarm Only"],
        "flood_zone_tier": TIER_OPTIONS, "earthquake_zone_tier": TIER_OPTIONS,
        "wildfire_zone_tier": TIER_OPTIONS, "wind_zone_tier": TIER_OPTIONS,
        "sprinklered_flag": ["Y", "N"],
    }
    industry_options = ["Accommodation & Food Services", "Administrative Services", "Construction", "Health Care",
                         "Manufacturing", "Professional Services", "Real Estate", "Retail Trade",
                         "Transportation & Warehousing", "Wholesale Trade"]
    tested, errors = 0, []
    for lob in PROPERTY_LOBS:
        base = base_row(lob, "mid")
        for field, options in dropdowns.items():
            for opt in options:
                tested += 1
                try:
                    s = score_one({**base, field: opt})
                    assert 0 <= s["loss_propensity_score_v2"] <= 100
                except Exception as e:
                    errors.append((lob, field, opt, str(e)))
    for lob in LIABILITY_LOBS:
        base = base_row(lob, "mid")
        for opt in industry_options:
            tested += 1
            try:
                score_one({**base, "industry_class_naics": opt})
            except Exception as e:
                errors.append((lob, "industry_class_naics", opt, str(e)))
    for lob in ALL_LOBS:
        base = base_row(lob, "mid")
        for tier in BROKER_TIERS:
            tested += 1
            try:
                score_one({**base, "broker_tier": tier})
            except Exception as e:
                errors.append((lob, "broker_tier", tier, str(e)))
    print(f"  {tested} combinations tested, {len(errors)} errors")
    for e in errors[:20]:
        print(" ", e)


def stage_6_boundaries():
    print("\n=== Stage 6: Numeric boundary values (min/max of every form field) ===")
    boundary_fields = {
        "year_built": [1900, 2026], "num_stories": [1, 50], "square_footage": [500, 500000],
        "total_insured_value_usd": [50000, 50000000], "roof_age_years": [0, 120],
        "fire_protection_class_iso_ppc": [1, 10], "number_of_employees": [1, 20000],
        "avg_wage_per_employee_usd": [15000, 300000], "financial_stability_score": [0, 100],
        "debt_to_equity_ratio": [0.0, 10.0], "claim_count_last_3_years": [0, 20],
        "large_loss_count_last_5_years": [0, 10], "broker_hit_bind_ratio_pct": [0, 100],
        "competing_quotes_count": [0, 10], "submission_completeness_pct": [0, 100],
    }
    tested, errors, out_of_range = 0, [], []
    for lob in ["Commercial Property", "Workers Compensation"]:
        base = base_row(lob, "mid")
        for field, (lo, hi) in boundary_fields.items():
            for v in [lo, hi]:
                tested += 1
                try:
                    s = score_one({**base, field: v})
                    for col in ["loss_propensity_score_v2", "bind_propensity_score_v2",
                                "appetite_capacity_score_v2", "uw_score_v2"]:
                        if not (0 <= s[col] <= 100):
                            out_of_range.append((lob, field, v, col, s[col]))
                except Exception as e:
                    errors.append((lob, field, v, str(e)))
    print(f"  {tested} boundary values tested, {len(errors)} crashes, {len(out_of_range)} scores outside 0-100")
    for e in errors: print(" ", e)
    for o in out_of_range: print(" ", o)


def stage_7_driver_ranking():
    print("\n=== Stage 7: Relative driver strength - Loss Propensity, ranked ===")
    base = base_row("Commercial Property", "mid")
    fields = {
        "total_insured_value_usd": (100000, 10000000), "year_built": (2023, 1930),
        "construction_type": ("Fire Resistive", "Frame"), "flood_zone_tier": ("Low", "Severe"),
        "sprinklered_flag": ("Y", "N"), "financial_stability_score": (95, 10),
        "claim_count_last_3_years": (0, 6), "roof_age_years": (2, 60),
        "debt_to_equity_ratio": (0.0, 5.0), "wind_zone_tier": ("Low", "Severe"),
    }
    results = []
    for field, (lo, hi) in fields.items():
        s1 = score_one({**base, field: lo})["loss_propensity_score_v2"]
        s2 = score_one({**base, field: hi})["loss_propensity_score_v2"]
        results.append((field, abs(s2 - s1)))
    results.sort(key=lambda x: -x[1])
    total = sum(r[1] for r in results)
    for field, swing in results:
        print(f"  {field:32s} swing={swing:6.1f}  ({swing/total*100:5.1f}% of tested swing)")


if __name__ == "__main__":
    stage_1_best_worst()
    stage_2_monotonicity()
    stage_3_bind_sensitivity()
    stage_4_appetite_gate()
    stage_5_dropdowns()
    stage_6_boundaries()
    stage_7_driver_ranking()
