"""
End-to-end model behavior test suite. Not part of the scoring pipeline itself - a
standalone check that everything still behaves correctly after any retrain or code change.
Run: python test_model_behavior.py
"""
import itertools
import pandas as pd
from scoring import load_models, score
from lob_config import PROPERTY_LOBS, LIABILITY_LOBS, CURRENT_YEAR, region_for_state
import driver_explain as dexp

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
        "primary_cat_peril_exposure": ["Low/Minimal CAT Exposure", "Wind/Hail", "Elevated — Multiple Perils"][lvl],
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
        # loss_propensity_score_v2 is clipped to [0,100] for display - for a couple of the
        # lowest-premium-rate LOBs (Professional Liability, Umbrella/Excess) both ends of this
        # synthetic best/worst scenario can genuinely exceed 100% loss ratio and tie at the
        # display ceiling even though the real underlying gap is large and correctly ordered
        # (found 2026-08: best=100.6%, worst=246.0% raw - a real, big, correct difference that
        # a strict "<" on the clipped score alone would falsely flag). Check the raw uncapped
        # ratio too so a genuine reversal is still caught, but a ceiling tie isn't a false alarm.
        loss_ok = (best["loss_propensity_score_v2"] < worst["loss_propensity_score_v2"]
                   or (best["loss_propensity_score_v2"] == worst["loss_propensity_score_v2"] == 100.0
                       and best["predicted_loss_ratio_pct"] < worst["predicted_loss_ratio_pct"]))
        ok = (loss_ok
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
        # added 2026-08 after the 8-LOB audit found these swinging the score non-monotonically
        # by up to 76 points with no constraint anywhere - now either constrained (should pass
        # cleanly) or correctly dropped by the "contradicts known direction" safeguard (flat,
        # which also passes cleanly) - either outcome is fine, erratic swinging is not
        "revenue_growth_trend_pct": [-30, -10, 0, 20, 60], "business_tenure_years": [0, 10, 25, 50, 80],
    }
    liability_sweeps = {
        "avg_weekly_work_hours": [18, 25, 32, 40, 50], "avg_employee_age": [19, 30, 45, 60, 66],
        "avg_wage_per_employee_usd": [46000, 55000, 70000, 85000, 94000],
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
        else:
            for field, values in liability_sweeps.items():
                total += 1
                scores_list = [score_one({**base, field: v})["loss_propensity_score_v2"] for v in values]
                if not (all(scores_list[i] <= scores_list[i+1] + 0.15 for i in range(len(scores_list)-1))
                        or all(scores_list[i] >= scores_list[i+1] - 0.15 for i in range(len(scores_list)-1))):
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
    # uses "best" risk (not "mid") deliberately - regression test for a real bug found
    # 2026-08: risk_tier_v2 used to be computed from uw_score_v2 alone, so a hard-rule
    # failure paired with an otherwise-clean loss/bind profile could still land in
    # Standard/High instead of Decline, contradicting the documented rule ("Decline: < 38,
    # OR any hard-rule fail"). "best" risk is the case most likely to expose this - genuinely
    # good loss/bind scores are exactly what could out-vote a zeroed appetite in the blend.
    base = base_row("Commercial Property", "best")
    flags = ["sanctions_ofac_watchlist_flag", "prior_insolvency_bankruptcy_flag",
             "restricted_operations_flag", "region_lob_restricted_flag"]
    violations, tier_violations = 0, []
    for combo in itertools.product([0, 1], repeat=4):
        r = {**base, **dict(zip(flags, combo))}
        s = score_one(r)
        if any(combo) and s["appetite_capacity_score_v2"] != 0:
            violations += 1
            print(f"  VIOLATION: flags={combo} -> Appetite={s['appetite_capacity_score_v2']}")
        if any(combo) and s["risk_tier_v2"] != "Decline":
            tier_violations.append((combo, s["risk_tier_v2"], s["uw_score_v2"]))
    print(f"  16/16 combinations tested, {violations} gate violations, {len(tier_violations)} risk_tier_v2 mismatches")
    for combo, tier, uw in tier_violations:
        print(f"    VIOLATION: flags={combo} -> risk_tier_v2={tier} (uw_score={uw}, should be Decline)")


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
        "primary_cat_peril_exposure": ["Low/Minimal CAT Exposure", "Flood/Storm Surge", "Wildfire",
                                        "Earthquake", "Wind/Hail", "Elevated — Multiple Perils"],
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
        "year_built": [1910, CURRENT_YEAR], "num_stories": [1, 11], "square_footage": [1000, 268000],
        "total_insured_value_usd": [40000, 25000000], "roof_age_years": [1, 120],
        "fire_protection_class_iso_ppc": [1, 10], "number_of_employees": [1, 5000],
        "avg_wage_per_employee_usd": [46000, 94000], "financial_stability_score": [5, 100],
        "debt_to_equity_ratio": [0.0, 5.0], "claim_count_last_3_years": [0, 9],
        "large_loss_count_last_5_years": [0, 3], "broker_hit_bind_ratio_pct": [0, 100],
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


def stage_8_unknown_category_safety():
    """Confirms a genuinely new category (never seen in training) still produces a real,
    usable score - no crash, no NaN. The app no longer surfaces a caveat for these cases
    (client-facing output shows the model's prediction plainly), but the underlying pipeline
    still has to hold up structurally regardless of what's typed into a live submission."""
    print("\n=== Stage 8: Unknown-category safety (new LOB / state / broker tier) ===")
    base = base_row("Commercial Property", "mid")
    cases = {
        "Unknown Line of Business": {**base, "line_of_business": "Cyber Liability (Not Yet Modeled)"},
        "Unknown State": {**base, "risk_state": "PR"},
        "Unknown Broker Tier": {**base, "broker_tier": "Diamond"},
        "Extreme out-of-range TIV ($750M)": {**base, "total_insured_value_usd": 750_000_000},
        "Unseen construction type": {**base, "construction_type": "Modular/Prefab"},
    }
    crashes, nan_scores = [], []
    for label, row in cases.items():
        try:
            s = score_one(row)
        except Exception as e:
            crashes.append((label, str(e)))
            continue
        for col in ["loss_propensity_score_v2", "bind_propensity_score_v2",
                    "appetite_capacity_score_v2", "uw_score_v2"]:
            if pd.isna(s[col]):
                nan_scores.append((label, col))
    print(f"  {len(cases)} unknown-category cases tested: "
          f"{len(crashes)} crashed, {len(nan_scores)} produced NaN")
    for c in crashes: print("  CRASH:", c)
    for n in nan_scores: print("  NaN SCORE:", n)


def stage_9_appetite_batch_independence():
    """Regression test for a real bug found 2026-08: appetite/capacity's industry-concentration
    headroom used to be computed against whatever dataframe was passed into score() itself, so
    a single live submission's own industry silently counted as 100% of the reference book -
    flooring the headroom credit to 0 on nearly every live score regardless of the real book's
    actual exposure. Fixed by referencing the real book's fixed INDUSTRY_CONCENTRATION_BY_NAICS
    constant instead. This confirms a submission's appetite score no longer depends on how many
    OTHER rows happen to be scored in the same batch - which is what "correct" looks like here."""
    print("\n=== Stage 9: Appetite/Capacity score independence from batch size ===")
    base = base_row("Commercial Property", "mid")
    base["industry_class_naics"] = "Real Estate"  # ~9.4% of the real book - comfortably under the 15% cap
    solo = score_one(base)["appetite_capacity_score_v2"]
    batch = pd.DataFrame([base] * 5)  # same row, scored together as a same-industry batch of 5
    batched = score(batch, models)["appetite_capacity_score_v2"].iloc[0]
    diff = abs(solo - batched)
    status = "PASS" if diff < 0.5 else "FAIL"
    print(f"  Solo score: {solo}, scored-in-batch-of-5 score: {batched}, diff={diff:.1f} ({status} - should be ~0)")


def stage_10_cat_tiv_penalty_fires():
    """Regression test for a real bug found 2026-08: scoring.py ordinal-encodes flood/EQ/
    wildfire tier (Low/Moderate/High/Severe -> 0-3) BEFORE calling the appetite score, but
    cat_tiv_penalty() compared against the string "Severe" - always False against an integer,
    so the penalty silently never fired for any live-scored submission. Confirms a severe-CAT
    + high-TIV submission now actually scores lower appetite than an identical one without it."""
    print("\n=== Stage 10: CAT/TIV penalty actually fires through the real pipeline ===")
    base = base_row("Commercial Property", "best")
    clean = score_one(base)["appetite_capacity_score_v2"]
    severe = score_one({**base, "flood_zone_tier": "Severe", "earthquake_zone_tier": "Severe",
                         "total_insured_value_usd": 5_000_000})["appetite_capacity_score_v2"]
    diff = clean - severe
    status = "PASS" if diff >= 8 else "FAIL"
    print(f"  Clean-zone appetite: {clean}, severe-zone+high-TIV appetite: {severe}, "
          f"penalty applied={diff:.1f} ({status} - should be ~9)")


def stage_11_decision_layer_consistency():
    """Regression test for the dashboard-wiring fix (2026-08): risk_tier_v2, the recommended
    action, and the appetite decision category must all agree with the numeric scores they're
    derived from - not the frozen original file columns the Executive Dashboard used to read
    before this fix (which never moved regardless of any scoring change)."""
    print("\n=== Stage 11: risk_tier_v2 / appetite_capacity_decision_v2 consistency ===")
    from decision_matrix_reference import tier_for_score, RISK_TIER_INFO
    issues = []
    for lob in ALL_LOBS:
        for level in ["best", "mid", "worst"]:
            r = score_one(base_row(lob, level))
            expected_tier = tier_for_score(r["uw_score_v2"])
            if r["risk_tier_v2"] != expected_tier:
                issues.append(f"{lob}/{level}: risk_tier_v2={r['risk_tier_v2']} but uw_score_v2="
                               f"{r['uw_score_v2']} implies {expected_tier}")
            if r["recommended_uw_action_v2"] != RISK_TIER_INFO[r["risk_tier_v2"]]["uw_action"]:
                issues.append(f"{lob}/{level}: recommended_uw_action_v2 doesn't match its own risk_tier_v2's action")
            if r["appetite_capacity_score_v2"] == 0 and r["appetite_capacity_decision_v2"] != "Decline – Outside Appetite":
                issues.append(f"{lob}/{level}: appetite=0 but decision={r['appetite_capacity_decision_v2']!r}")
    print(f"  {24} checks run, {len(issues)} inconsistencies")
    for i in issues:
        print("   ", i)


def stage_12_exposure_unit_scaling():
    """Regression test for the exposure-unit fix (2026-08): frequency_model.py used to treat
    number_of_employees/total_insured_value_usd as ordinary features rather than an explicit
    exposure multiplier, so predicted frequency barely moved across their entire real range
    while premium scaled linearly with them - collapsing loss ratio toward zero at scale for
    purely mechanical reasons (a 5,000-employee WC account scored 0.0, essentially risk-free).
    Confirms predicted_frequency now scales up with size, and a large account no longer reads
    as near-zero risk."""
    print("\n=== Stage 12: Exposure-unit scaling (size no longer collapses risk to ~0) ===")
    issues = []
    for lob in LIABILITY_LOBS:
        base = base_row(lob, "mid")
        small = score_one({**base, "number_of_employees": 5})
        large = score_one({**base, "number_of_employees": 2000})
        if not (small["predicted_frequency"] < large["predicted_frequency"]):
            issues.append(f"{lob}: predicted_frequency did not increase with employee count "
                           f"({small['predicted_frequency']} -> {large['predicted_frequency']})")
        if large["loss_propensity_score_v2"] < 3.0:
            issues.append(f"{lob}: a 2,000-employee account still scores near-zero risk "
                           f"({large['loss_propensity_score_v2']}) - exposure scaling may have regressed")
    for lob in PROPERTY_LOBS:
        base = base_row(lob, "mid")
        small = score_one({**base, "total_insured_value_usd": 200000})
        large = score_one({**base, "total_insured_value_usd": 15000000})
        if not (small["predicted_frequency"] < large["predicted_frequency"]):
            issues.append(f"{lob}: predicted_frequency did not increase with TIV "
                           f"({small['predicted_frequency']} -> {large['predicted_frequency']})")
        if large["loss_propensity_score_v2"] < 3.0:
            issues.append(f"{lob}: a $15M-TIV account still scores near-zero risk "
                           f"({large['loss_propensity_score_v2']}) - exposure scaling may have regressed")
    print(f"  {len(ALL_LOBS) * 2} checks run, {len(issues)} issues")
    for i in issues:
        print("   ", i)


def stage_13_direct_channel_no_broker():
    """Regression test for the Direct-channel/no-broker business rule (2026-08): Direct means
    no broker relationship at all, and the What-If Simulator now leaves broker_tier unset for
    a Direct submission rather than asking for one. Confirms (a) that doesn't crash Bind
    Propensity to its 0 floor - dropping the 7 broker-detail terms outright would, since
    BIND_COEFFICIENTS' large negative intercept assumes they always contribute something
    close to a real book-typical value (see bind_propensity.py's populate_broker_fields
    docstring) - and (b) a Broker-channel submission's score is bit-for-bit unaffected by
    this change."""
    print("\n=== Stage 13: Direct channel / no-broker business rule ===")
    issues = []
    for lob in ALL_LOBS:
        base = base_row(lob, "mid")
        broker_row = {**base, "channel": "Broker"}
        direct_row = {**base, "channel": "Direct", "broker_tier": None}

        broker_result = score_one(broker_row)
        direct_result = score_one(direct_row)

        if direct_result["bind_propensity_score_v2"] <= 5.0:
            issues.append(f"{lob}: Direct-channel Bind Propensity floored to "
                           f"{direct_result['bind_propensity_score_v2']} - broker terms likely "
                           f"dropped outright instead of using the neutral fallback profile")
        # Direct channel carries its own real, fitted bonus (+5.98 vs. Broker reference) - a
        # Direct submission with an otherwise-identical profile should never score LOWER on
        # Bind than the same submission through a Broker.
        if direct_result["bind_propensity_score_v2"] < broker_result["bind_propensity_score_v2"] - 0.5:
            issues.append(f"{lob}: Direct ({direct_result['bind_propensity_score_v2']}) scored "
                           f"lower than Broker ({broker_result['bind_propensity_score_v2']}) - "
                           f"the Direct-channel bonus may have regressed")

        # A plain Broker-channel row (this stage's own baseline) must still match what stage 1-12
        # already expect elsewhere - loss/appetite scores shouldn't move at all from a channel-only change.
        broker_row_no_channel = {**base}
        no_channel_result = score_one(broker_row_no_channel)
        if abs(broker_result["uw_score_v2"] - no_channel_result["uw_score_v2"]) > 0.05:
            issues.append(f"{lob}: explicit channel='Broker' scored differently from omitting "
                           f"channel entirely ({broker_result['uw_score_v2']} vs "
                           f"{no_channel_result['uw_score_v2']}) - Broker should be the default")
    print(f"  {len(ALL_LOBS)} lines checked, {len(issues)} issues")
    for i in issues:
        print("   ", i)


def stage_14b_loss_bind_driver_direction():
    """Same audit as stage_14, extended to Loss and Bind after a client-reported bug in
    Appetite's driver direction (2026-10) prompted checking the other two panels too - they
    turned out correct, but had no permanent regression guard either. Scoped to a handful of
    fields with an unambiguous, model-independent expected direction (claims history,
    financial stability, roof age, broker tier) - deliberately excludes fields base_row()
    holds constant across levels (number_of_employees, avg_wage_per_employee_usd,
    industry_class_naics - these showed up as false alarms in the investigation behind this
    stage, same root cause as stage_14's Industry Concentration exclusion) and
    construction_type (verified against the real book: Fire Resistive has the lowest claim
    frequency but NOT the lowest severity of any class - a genuine, data-driven nuance from
    deliberately leaving this categorical field unconstrained, not a bug to encode here as if
    there were one unambiguous right answer)."""
    print("\n=== Stage 14b: Loss + Bind driver direction sanity (favorable direction) ===")
    issues = []
    checked = 0
    loss_expect_favorable_field = {"Claims-Free Years", "Claim Count (3yr)", "Financial Stability Score",
                                    "Roof Age", "Business Tenure", "Large-Loss Count (5yr)"}
    for lob in ALL_LOBS:
        for level, want_favorable in [("best", True), ("worst", False)]:
            row_dict = base_row(lob, level)
            row_dict["risk_location_region"] = region_for_state(row_dict["risk_state"])
            row_df = pd.DataFrame([row_dict])

            loss_drv = dexp.loss_drivers(row_df, lambda d: score(d, models).iloc[0])
            for d in loss_drv:
                if d["label"] not in loss_expect_favorable_field:
                    continue
                checked += 1
                is_favorable = d["direction"] == "decreases"
                if is_favorable != want_favorable:
                    issues.append(f"{lob}/{level} LOSS: {d['label']} dir={d['direction']} "
                                   f"(expected {'favorable' if want_favorable else 'unfavorable'})")

            bind_drv = dexp.bind_drivers(row_dict)
            for d in bind_drv:
                if not d["label"].startswith("Broker Tier"):
                    continue
                checked += 1
                is_favorable = d["direction"] == "increases"
                if is_favorable != want_favorable:
                    issues.append(f"{lob}/{level} BIND: {d['label']} dir={d['direction']} "
                                   f"(expected {'favorable' if want_favorable else 'unfavorable'})")
    print(f"  {checked} driver entries checked, {len(issues)} issues")
    for i in issues:
        print("   ", i)


def stage_14_appetite_driver_direction():
    """Regression test for a real bug (2026-10, caught via a client screenshot): appetite_drivers()
    used to flip the sign of every single driver it returned, so a genuinely favorable field
    (e.g. Financial Stability Score well above the book average) rendered as a red/unfavorable
    bar in the UI, and a genuinely unfavorable field would have rendered green - every driver
    in the panel was backwards, not just one. Confirms a "best" scenario's drivers now report
    the favorable direction and a "worst" scenario's report the unfavorable one."""
    print("\n=== Stage 14: Appetite driver direction sanity (favorable = 'increases') ===")
    issues = []
    for lob in ALL_LOBS:
        for level, want_favorable in [("best", True), ("worst", False)]:
            row = base_row(lob, level)
            row["risk_location_region"] = region_for_state(row["risk_state"])
            drv = dexp.appetite_drivers(row)
            for d in drv:
                # industry_class_naics is a fixed constant in base_row() ("Manufacturing"),
                # never varied by level - and Manufacturing genuinely sits over the real 15%
                # concentration cap (16.36%) regardless of how favorable everything else is,
                # so its driver correctly stays unfavorable even in a "best" scenario. Not a
                # bug - skip it here rather than asserting something that was never true.
                if d["label"].startswith("Industry Concentration"):
                    continue
                is_favorable = d["direction"] == "increases"
                if is_favorable != want_favorable:
                    issues.append(f"{lob}/{level}: '{d['label']}' reported direction="
                                   f"'{d['direction']}' ({'favorable' if is_favorable else 'unfavorable'}) "
                                   f"but this is a {level} scenario - expected the opposite")
    print(f"  {len(ALL_LOBS) * 2} scenarios checked, {len(issues)} issues")
    for i in issues:
        print("   ", i)


if __name__ == "__main__":
    stage_1_best_worst()
    stage_2_monotonicity()
    stage_3_bind_sensitivity()
    stage_4_appetite_gate()
    stage_5_dropdowns()
    stage_6_boundaries()
    stage_7_driver_ranking()
    stage_8_unknown_category_safety()
    stage_9_appetite_batch_independence()
    stage_10_cat_tiv_penalty_fires()
    stage_11_decision_layer_consistency()
    stage_12_exposure_unit_scaling()
    stage_13_direct_channel_no_broker()
    stage_14_appetite_driver_direction()
    stage_14b_loss_bind_driver_direction()
