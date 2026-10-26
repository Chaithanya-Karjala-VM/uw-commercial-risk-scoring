const TIER_OPTIONS = ["Low", "Moderate", "High", "Severe"];

async function initWhatIf() {
    const root = document.getElementById("page-whatif");
    const filters = await apiGet("/api/filters");
    const field = (label, inner) => `<div class="wi-field"><label>${label}</label>${inner}</div>`;
    root.innerHTML = `
        <div class="card">
            <div class="eyebrow">What-If Simulator - scores a real scenario through the exact same model as the rest of this app</div>
            <form id="wi-form">
                <div class="wi-top-row">
                    ${field("Line of Business", `<select name="line_of_business">${filters.lobs.map((l) => `<option>${escapeHtml(l)}</option>`).join("")}</select>`)}
                    ${field("State", `<input type="text" name="risk_state" value="CA" maxlength="2">`)}
                    ${field("Channel", `<select name="channel"><option>Broker</option><option>Direct</option><option>MGA</option></select>`)}
                    ${field("Renewal / New", `<select name="renewal_flag"><option>New Business</option><option>Renewal</option></select>`)}
                    ${field("Industry (NAICS)", `<select name="industry_class_naics">
                        ${["Accommodation & Food Services","Administrative Services","Construction","Health Care","Manufacturing","Professional Services","Real Estate","Retail Trade","Transportation & Warehousing","Wholesale Trade"].map((x) => `<option>${x}</option>`).join("")}
                        </select>`)}
                </div>

                <div class="wi-form-grid">
                    <div class="wi-group">
                        <div class="wi-group-title">Property</div>
                        ${field("Construction Type", `<select name="construction_type">${["Fire Resistive","Frame","Joisted Masonry","Non-Combustible","Masonry Non-Combustible"].map((x) => `<option>${x}</option>`).join("")}</select>`)}
                        ${field("Year Built", `<input type="number" name="year_built" value="2000" min="1910" max="2026">`)}
                        ${field("Number of Stories", `<input type="number" name="num_stories" value="2" min="1" max="11">`)}
                        ${field("Square Footage", `<input type="number" name="square_footage" value="5000" min="1000">`)}
                        ${field("Total Insured Value ($)", `<input type="number" name="total_insured_value_usd" value="500000" min="40000" step="10000">`)}
                        ${field("Occupancy Type", `<input type="text" name="occupancy_type" value="Office">`)}
                        ${field("Roof Age (yrs)", `<input type="number" name="roof_age_years" value="15" min="1" max="120">`)}
                        ${field("Sprinklered", `<select name="sprinklered_flag"><option value="Y">Yes</option><option value="N">No</option></select>`)}
                    </div>

                    <div class="wi-group">
                        <div class="wi-group-title">Catastrophe exposure</div>
                        ${field("Flood Zone", `<select name="flood_zone_tier">${TIER_OPTIONS.map((t) => `<option>${t}</option>`).join("")}</select>`)}
                        ${field("Earthquake Zone", `<select name="earthquake_zone_tier">${TIER_OPTIONS.map((t) => `<option>${t}</option>`).join("")}</select>`)}
                        ${field("Wildfire Zone", `<select name="wildfire_zone_tier">${TIER_OPTIONS.map((t) => `<option>${t}</option>`).join("")}</select>`)}
                        ${field("Wind Zone", `<select name="wind_zone_tier">${TIER_OPTIONS.map((t) => `<option>${t}</option>`).join("")}</select>`)}
                        ${field("Distance to Coast (mi)", `<input type="number" name="distance_to_coast_miles" value="60" step="0.1">`)}
                    </div>

                    <div class="wi-group">
                        <div class="wi-group-title">Claims &amp; financial history</div>
                        ${field("Claim Count (3yr)", `<input type="number" name="claim_count_last_3_years" value="0" min="0" max="9">`)}
                        ${field("Claims-Free Years", `<input type="number" name="claims_free_years" value="5" min="0" max="10">`)}
                        ${field("Large-Loss Count (5yr)", `<input type="number" name="large_loss_count_last_5_years" value="0" min="0" max="3">`)}
                        ${field("Financial Stability Score", `<input type="number" name="financial_stability_score" value="65" min="5" max="100">`)}
                        ${field("Safety Program", `<select name="safety_program_flag"><option value="true">Yes</option><option value="false">No</option></select>`)}
                        ${field("Business Tenure (yrs)", `<input type="number" name="business_tenure_years" value="10" min="0" max="80">`)}
                        ${field("Revenue Growth (%)", `<input type="number" name="revenue_growth_trend_pct" value="3" step="0.1">`)}
                        ${field("Debt-to-Equity", `<input type="number" name="debt_to_equity_ratio" value="1.0" step="0.1" min="0" max="5">`)}
                    </div>

                    <div class="wi-group">
                        <div class="wi-group-title">Broker &amp; deal</div>
                        ${field("Broker Tier", `<select name="broker_tier"><option>Platinum</option><option selected>Gold</option><option>Silver</option><option>Bronze</option></select>`)}
                        ${field("Competing Quotes", `<input type="number" name="competing_quotes_count" value="2" min="0" max="10">`)}
                        ${field("Coverage Limit ($)", `<input type="number" name="coverage_limit_requested_usd" value="500000" min="50000" step="10000">`)}
                        ${field("Policy Term (months)", `<select name="policy_term_months"><option>6</option><option selected>12</option><option>24</option><option>36</option></select>`)}
                    </div>

                    <div class="wi-group">
                        <div class="wi-group-title">Compliance</div>
                        ${field("Sanctions/OFAC Hit", `<select name="sanctions_ofac_watchlist_flag"><option value="0">No</option><option value="1">Yes</option></select>`)}
                        ${field("Prior Insolvency", `<select name="prior_insolvency_bankruptcy_flag"><option value="0">No</option><option value="1">Yes</option></select>`)}
                        ${field("Restricted Operations", `<select name="restricted_operations_flag"><option value="0">No</option><option value="1">Yes</option></select>`)}
                        ${field("Region x LOB Restricted", `<select name="region_lob_restricted_flag"><option value="0">No</option><option value="1">Yes</option></select>`)}
                    </div>
                </div>

                <button type="submit" class="btn btn-primary" style="margin-top:22px;">Calculate</button>
            </form>
        </div>
        <div id="wi-result"></div>
    `;
    document.getElementById("wi-form").addEventListener("submit", onWhatIfSubmit);
}

function buildWhatIfPayload(form) {
    const fd = new FormData(form);
    const yearBuilt = parseInt(fd.get("year_built"), 10);
    return {
        line_of_business: fd.get("line_of_business"), risk_state: fd.get("risk_state"),
        channel: fd.get("channel"), renewal_flag: fd.get("renewal_flag"), industry_class_naics: fd.get("industry_class_naics"),
        construction_type: fd.get("construction_type"), year_built: yearBuilt,
        building_age_years: new Date().getFullYear() - yearBuilt,
        built_to_current_code_flag: true, num_stories: parseInt(fd.get("num_stories"), 10),
        square_footage: parseInt(fd.get("square_footage"), 10), total_insured_value_usd: parseFloat(fd.get("total_insured_value_usd")),
        occupancy_type: fd.get("occupancy_type"), roof_age_years: parseInt(fd.get("roof_age_years"), 10),
        roof_type: "Composition Shingle", sprinklered_flag: fd.get("sprinklered_flag"), fire_protection_class_iso_ppc: 5,
        distance_to_fire_station_miles: 5.0, distance_to_fire_hydrant_miles: 0.5, fire_alarm_monitoring_type: "Central Station",
        flood_zone_tier: fd.get("flood_zone_tier"), fema_flood_zone_code: "X", earthquake_zone_tier: fd.get("earthquake_zone_tier"),
        soil_liquefaction_class: "Low", wildfire_zone_tier: fd.get("wildfire_zone_tier"), wind_zone_tier: fd.get("wind_zone_tier"),
        primary_cat_peril_exposure: "None/Minimal", distance_to_coast_miles: parseFloat(fd.get("distance_to_coast_miles")),
        distance_to_fault_line_miles: 60.0, distance_to_nearest_water_body_miles: 3.0,
        number_of_employees: 25, avg_wage_per_employee_usd: 55000, avg_employee_age: 40, avg_weekly_work_hours: 40.0,
        claim_count_last_3_years: parseInt(fd.get("claim_count_last_3_years"), 10),
        large_loss_count_last_5_years: parseInt(fd.get("large_loss_count_last_5_years"), 10),
        claims_free_years: parseInt(fd.get("claims_free_years"), 10),
        financial_stability_score: parseFloat(fd.get("financial_stability_score")),
        safety_program_flag: fd.get("safety_program_flag") === "true", safety_inspection_findings: "No Issues Noted",
        business_tenure_years: parseFloat(fd.get("business_tenure_years")),
        revenue_growth_trend_pct: parseFloat(fd.get("revenue_growth_trend_pct")),
        debt_to_equity_ratio: parseFloat(fd.get("debt_to_equity_ratio")),
        policy_term_months: parseInt(fd.get("policy_term_months"), 10), broker_tier: fd.get("broker_tier"),
        competing_quotes_count: parseInt(fd.get("competing_quotes_count"), 10),
        coverage_limit_requested_usd: parseFloat(fd.get("coverage_limit_requested_usd")),
        sanctions_ofac_watchlist_flag: parseInt(fd.get("sanctions_ofac_watchlist_flag"), 10),
        prior_insolvency_bankruptcy_flag: parseInt(fd.get("prior_insolvency_bankruptcy_flag"), 10),
        restricted_operations_flag: parseInt(fd.get("restricted_operations_flag"), 10),
        region_lob_restricted_flag: parseInt(fd.get("region_lob_restricted_flag"), 10),
        target_growth_segment_flag: 0,
    };
}

async function onWhatIfSubmit(e) {
    e.preventDefault();
    const resultEl = document.getElementById("wi-result");
    resultEl.innerHTML = `<div class="loading">Scoring...</div>`;
    try {
        const payload = buildWhatIfPayload(e.target);
        const result = await apiPost("/api/score", payload);
        renderWhatIfResult(result);
    } catch (err) {
        resultEl.innerHTML = `<div class="card" style="color:var(--bad);">Scoring failed: ${escapeHtml(err.message)}</div>`;
    }
}

function wiCell(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "boolean") return v ? "True" : "False";
    return String(v);
}

function wiCalcRows(t) {
    return [
        ["Hard Stop", t.hard_stop],
        ["Line of Business", t.line_of_business],
        ["Predicted Claim Frequency", t.predicted_frequency],
        ["Predicted Attritional Severity ($)", t.predicted_attritional_severity],
        ["Probability of Large Loss", t.p_large_loss],
        ["Expected Large-Loss Severity ($)", t.expected_large_severity],
        ["Total Expected Loss Cost ($)", t.total_expected_loss_cost],
        ["Predicted Loss Ratio (%)", t.predicted_loss_ratio_pct],
        ["Loss Propensity Score", t.loss_propensity_score],
        ["Bind Propensity Score", t.bind_propensity_score],
        ["Appetite/Capacity Score", t.appetite_capacity_score],
        ["Composite UW Score", t.composite_uw_score],
        ["Risk Tier", t.risk_tier],
        ["Appetite Decision", t.appetite_decision],
        ["Recommended Action", t.recommended_action],
    ];
}

function wiReasonCodesHtml(lossDrv, bindDrv, appetiteDrv) {
    const rows = [
        ...lossDrv.map((d) => ({ cat: "Loss", ...d })),
        ...bindDrv.map((d) => ({ cat: "Bind", ...d })),
        ...appetiteDrv.map((d) => ({ cat: "Appetite", ...d })),
    ];
    if (!rows.length) return `<div class="caption">No material drivers to rank for this submission.</div>`;
    return `<table class="data-table"><thead><tr><th>Category</th><th>Driver</th><th>Direction</th><th>Relative Weight</th></tr></thead><tbody>
        ${rows.map((r) => `<tr><td>${escapeHtml(r.cat)}</td><td>${escapeHtml(r.label)}</td><td>${escapeHtml(r.direction)}</td><td>${fmtNum(r.weight)}</td></tr>`).join("")}
    </tbody></table>`;
}

function renderWhatIfResult(result) {
    const s = result.scores;
    const t = result.technical;
    const lossDrv = result.drivers.loss, bindDrv = result.drivers.bind, appetiteDrv = result.drivers.appetite;

    const bandOrder = [["Decline", 0, 38], ["High", 38, 55], ["Standard", 55, 72], ["Preferred", 72, 100]];
    const triageBar = bandOrder.map(([name, lo, hi]) =>
        `<div style="flex:${hi - lo}; background:${TIER_COLORS[name]}; height:22px; display:flex; align-items:center; justify-content:center; color:white; font-size:0.68rem; font-weight:700;">${name} (${lo}-${hi})</div>`).join("");
    const markerPct = Math.min(Math.max(s.uw_score, 0), 100);

    // Top 2 drivers from each category, deduped - same construction as the Streamlit
    // banner's "Key Factors Behind This Decision" chip row.
    const chipSource = [...lossDrv.slice(0, 2), ...bindDrv.slice(0, 2), ...appetiteDrv.slice(0, 2)];
    const seen = new Set(), chips = [];
    for (const d of chipSource) {
        if (!seen.has(d.label)) { chips.push(d.label); seen.add(d.label); }
    }

    const divergeCol = (drv, goodDirection, emptyMsg) => drv.length
        ? drv.map((d) => divergingBarHtml(d, goodDirection)).join("")
        : `<div class="caption">${emptyMsg}</div>`;

    document.getElementById("wi-result").innerHTML = `
        <div class="eyebrow" style="margin-top:4px;">Underwriting result</div>
        <div class="card banner" style="border-left-color:${TIER_COLORS[s.risk_tier]}; background:color-mix(in srgb, ${TIER_COLORS[s.risk_tier]} 10%, var(--card-bg));">
            <div style="font-size:1.5rem; font-weight:800; color:${TIER_COLORS[s.risk_tier]};">${escapeHtml(s.risk_tier)}</div>
            <div style="margin-top:4px;"><b>Recommended Action:</b> ${escapeHtml(s.recommended_action)}</div>
            <div class="caption" style="margin-top:6px;">${escapeHtml(s.banner_summary || result.narrative)}</div>
            <div style="margin-top:14px; position:relative;">
                <div style="display:flex; border-radius:6px; overflow:hidden;">${triageBar}</div>
                <div style="position:absolute; left:${markerPct}%; top:-4px; width:3px; height:30px; background:var(--text-main);"></div>
            </div>
            <div class="caption" style="margin-top:4px;">UW Score: ${fmtNum(s.uw_score)} / 100</div>
        </div>

        <div class="kpi-row">
            <div class="kpi-card"><div class="label">Composite UW Score</div><div class="value">${fmtNum(s.uw_score)}/100</div></div>
            <div class="kpi-card"><div class="label">Loss Propensity Score</div><div class="value">${fmtNum(s.loss_propensity)}</div></div>
            <div class="kpi-card"><div class="label">Bind Propensity Score</div><div class="value">${fmtNum(s.bind_propensity)}</div></div>
            <div class="kpi-card"><div class="label">Appetite Score</div><div class="value">${fmtNum(s.appetite_capacity)}</div></div>
        </div>

        ${chips.length ? `<div class="card">
            <div class="eyebrow">Key factors behind this decision</div>
            ${chips.map((c) => `<span class="ingredient-chip">${escapeHtml(c)}</span>`).join("")}
        </div>` : ""}

        <div class="card">
            <div class="eyebrow">Why this score</div>
            <div class="caption" style="margin-bottom:8px;">Top factors behind each score, measured against the real book's own typical submission. Green pulls toward Preferred, red pulls toward Decline.</div>
            <div class="three-col">
                <div>
                    <div class="sub-label">Loss Risk Drivers</div>
                    ${divergeCol(lossDrv, "decreases", "No material drivers vs. book baseline.")}
                </div>
                <div>
                    <div class="sub-label">Bind Propensity Drivers</div>
                    ${divergeCol(bindDrv, "increases", "No material drivers vs. book baseline.")}
                </div>
                <div>
                    <div class="sub-label">Appetite / Eligibility Drivers</div>
                    ${divergeCol(appetiteDrv, "increases", "Clean appetite profile.")}
                </div>
            </div>
        </div>

        <div class="card">
            <div class="eyebrow">Narrative</div>
            <div class="narrative-box">${escapeHtml(result.narrative)}</div>
        </div>

        <div class="card">
            <div style="display:flex; justify-content:space-between; align-items:center; cursor:pointer;" id="wi-tech-toggle">
                <div class="eyebrow" style="margin-bottom:0;">Technical &amp; audit detail (click to expand)</div>
                <span class="caption">&#9660;</span>
            </div>
            <div id="wi-tech-body" style="display:none; margin-top:10px;">
                <div class="caption" style="margin-bottom:10px;">Every figure below is deterministic and reproducible. The same inputs always produce the same output. The narrative above only explains these numbers in plain English; it never alters them.</div>
                <div class="btn-row" id="wi-tech-tabs" style="margin-bottom:10px;">
                    <button type="button" class="btn active-toggle" data-tab="calc">Full calculation</button>
                    <button type="button" class="btn" data-tab="reason">Reason codes</button>
                </div>
                <div id="wi-tech-calc">
                    <table class="data-table"><thead><tr><th>Field</th><th>Value</th></tr></thead><tbody>
                        ${wiCalcRows(t).map(([k, v]) => `<tr><td>${escapeHtml(k)}</td><td>${escapeHtml(wiCell(v))}</td></tr>`).join("")}
                    </tbody></table>
                </div>
                <div id="wi-tech-reason" style="display:none;">
                    ${wiReasonCodesHtml(lossDrv, bindDrv, appetiteDrv)}
                </div>
            </div>
        </div>
    `;

    document.getElementById("wi-tech-toggle").addEventListener("click", () => {
        const body = document.getElementById("wi-tech-body");
        body.style.display = body.style.display === "none" ? "block" : "none";
    });
    document.getElementById("wi-tech-tabs").querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.getElementById("wi-tech-tabs").querySelectorAll("button").forEach((b) => b.classList.remove("active-toggle"));
            btn.classList.add("active-toggle");
            document.getElementById("wi-tech-calc").style.display = btn.dataset.tab === "calc" ? "block" : "none";
            document.getElementById("wi-tech-reason").style.display = btn.dataset.tab === "reason" ? "block" : "none";
        });
    });
}
