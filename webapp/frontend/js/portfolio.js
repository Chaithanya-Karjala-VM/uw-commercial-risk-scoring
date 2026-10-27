const portfolioState = { lob: "All", region: "All", tier: "All", broker_tier: "All", renewal: "All", gridMetric: "uw_score" };

async function initPortfolio() {
    const root = document.getElementById("page-portfolio");
    root.innerHTML = `
        <div class="card">
            <div class="filter-row" id="pf-filters"></div>
        </div>

        <div class="card">
            <div class="eyebrow">How every submission is assessed</div>
            <div id="pf-flow"></div>
        </div>

        <div class="eyebrow" style="margin-top:4px;">Portfolio KPIs</div>
        <div id="pf-kpis" class="kpi-row"></div>

        <div class="card"><div class="mech-title">Triage outcome &mdash; risk tier (composite)</div>
            <div id="pf-tier-bar" style="margin:4px 0 6px;"></div>
            <div id="pf-tier-legend" style="margin-bottom:16px;"></div>
            <div id="pf-tier-mix"></div></div>

        <div class="two-col">
            <div class="card"><div class="mech-title">Appetite / capacity decision</div><div id="pf-appetite"></div></div>
            <div class="card"><div class="mech-title" style="color:var(--bad);">Why submissions fail hard rules (first rule failed)</div><div id="pf-hardfail-reasons"></div></div>
        </div>

        <div class="two-col">
            <div class="card">
                <div class="mech-title">UW score distribution, with triage bands shaded</div>
                <div id="pf-uw-dist"></div>
            </div>
            <div class="card">
                <div class="mech-title">Risk&ndash;Opportunity Quadrant Map</div>
                <div id="pf-quadrant"></div>
            </div>
        </div>

        <div class="card">
            <div class="mech-title">Line of business &mdash; premium, loss ratio, avg UW score, tier mix</div>
            <div id="pf-lob-table"></div>
        </div>

        <div class="card">
            <div class="mech-title">Where to look &mdash; average score by LOB x risk region</div>
            <div class="caption" style="margin-bottom:10px;">Click a cell to filter every chart on this page to that Line of Business x Region combination. Click it again to clear.</div>
            <div class="btn-row" id="pf-grid-metric-toggle" style="margin-bottom:12px;"></div>
            <div id="pf-grid"></div>
        </div>

        <div class="two-col">
            <div class="card"><div class="mech-title">Exposure &mdash; industry share of book TIV vs 15% cap</div><div id="pf-industry"></div></div>
            <div class="card"><div class="mech-title">Reinsurance treaties over limit (headroom)</div><div id="pf-treaties"></div></div>
        </div>
        <div class="card"><div class="mech-title">Broker tier &mdash; bind and UW score</div><div id="pf-broker-tier"></div></div>
    `;
    renderPfFlow();
    renderPortfolioFilters();
    await refreshPortfolio();
}

function renderPfFlow() {
    const box = "height:150px; box-sizing:border-box; border-radius:10px; display:flex; flex-direction:column; justify-content:center; background:var(--page-bg); border:1px solid var(--card-border);";
    const mini = "background:var(--card-bg); border:1px solid var(--card-border); border-radius:7px; padding:5px 9px; margin-bottom:5px;";
    const arrow = `<div style="color:var(--navy); font-size:1.5rem; font-weight:300; padding:0 2px; align-self:center;">&rarr;</div>`;
    document.getElementById("pf-flow").innerHTML = `
    <div style="display:flex; align-items:stretch; gap:8px; overflow-x:auto; padding:4px 0;">
        <div style="${box} padding:12px 16px; min-width:140px; align-items:center; text-align:center;">
            <div style="color:var(--navy); font-weight:800; font-size:0.84rem;">SUBMISSION<br>RECEIVED</div>
        </div>
        ${arrow}
        <div style="${box} padding:10px 14px; min-width:260px;">
            <div style="${mini}"><b style="color:var(--bad); font-size:0.76rem;">LOSS SCORE</b><br><span class="caption" style="font-size:0.7rem;">expected loss &mdash; 0 best / 100 worst</span></div>
            <div style="${mini}"><b style="color:var(--good); font-size:0.76rem;">APPETITE SCORE</b><br><span class="caption" style="font-size:0.7rem;">fit with appetite &amp; capacity</span></div>
            <div style="${mini}; margin-bottom:0;"><b style="color:var(--accent); font-size:0.76rem;">BIND SCORE</b><br><span class="caption" style="font-size:0.7rem;">likelihood we win it</span></div>
        </div>
        ${arrow}
        <div style="${box} padding:12px 16px; min-width:170px; align-items:center; text-align:center;">
            <div style="color:var(--navy); font-weight:800; font-size:0.84rem;">UW SCORE</div>
            <div class="caption" style="font-size:0.7rem; margin-top:5px; line-height:1.5;">0&ndash;100, weighted blend<br>50% Loss / 30% Appetite / 20% Bind</div>
        </div>
        ${arrow}
        <div style="${box} padding:12px 16px; min-width:250px;">
            <div style="color:var(--navy); font-weight:800; font-size:0.78rem; margin-bottom:7px; text-align:center;">TRIAGE</div>
            <div style="display:flex; border-radius:6px; overflow:hidden;">
                <div style="flex:1; background:var(--good); color:white; text-align:center; font-size:0.64rem; font-weight:700; padding:7px 2px;">Preferred</div>
                <div style="flex:1; background:var(--accent); color:white; text-align:center; font-size:0.64rem; font-weight:700; padding:7px 2px;">Standard</div>
                <div style="flex:1; background:var(--warn); color:white; text-align:center; font-size:0.64rem; font-weight:700; padding:7px 2px;">High</div>
                <div style="flex:1; background:var(--bad); color:white; text-align:center; font-size:0.64rem; font-weight:700; padding:7px 2px;">Decline</div>
            </div>
        </div>
        ${arrow}
        <div style="${box} padding:12px 16px; min-width:150px; align-items:center; text-align:center;">
            <div style="color:var(--navy); font-weight:800; font-size:0.78rem;">DRIVERS</div>
            <div class="caption" style="font-size:0.7rem; margin-top:5px;">Top reasons behind each score</div>
        </div>
    </div>`;
}

function renderPortfolioFilters() {
    apiGet("/api/filters").then((f) => {
        const el = document.getElementById("pf-filters");
        const mk = (key, label, options) => `
            <div class="filter-item"><label class="field-label">${label}</label>
            <select id="pf-filter-${key}">
                <option value="All">All</option>
                ${options.map((o) => `<option value="${escapeHtml(o)}">${escapeHtml(o)}</option>`).join("")}
            </select></div>`;
        el.innerHTML = mk("lob", "Line of Business", f.lobs) + mk("region", "Region", f.regions)
            + mk("tier", "Risk Tier", f.tiers) + mk("broker_tier", "Broker Tier", f.broker_tiers)
            + mk("renewal", "Renewal / New Business", f.renewal);
        ["lob", "region", "tier", "broker_tier", "renewal"].forEach((key) => {
            document.getElementById(`pf-filter-${key}`).addEventListener("change", (e) => {
                portfolioState[key] = e.target.value;
                refreshPortfolio();
            });
        });
    });
}

async function refreshPortfolio() {
    const p = { lob: portfolioState.lob, region: portfolioState.region, tier: portfolioState.tier,
                broker_tier: portfolioState.broker_tier, renewal: portfolioState.renewal };
    // Keep selects in sync when a grid click changes portfolioState programmatically.
    ["lob", "region", "tier", "broker_tier", "renewal"].forEach((key) => {
        const el = document.getElementById(`pf-filter-${key}`);
        if (el && el.value !== portfolioState[key]) el.value = portfolioState[key];
    });
    const [kpis, tierMix, appetite, uwDist, quadrant, lobTable, grid, industry, treaties, brokerTier] = await Promise.all([
        apiGet("/api/portfolio/kpis", p), apiGet("/api/portfolio/tier-mix", p), apiGet("/api/portfolio/appetite-decision", p),
        apiGet("/api/portfolio/uw-distribution", p), apiGet("/api/portfolio/quadrant", p), apiGet("/api/portfolio/lob-table", p),
        apiGet("/api/portfolio/grid", { ...p, metric: portfolioState.gridMetric }),
        apiGet("/api/portfolio/industry-concentration", p), apiGet("/api/portfolio/treaties", p),
        apiGet("/api/portfolio/broker-tier", p),
    ]);
    renderPfKpis(kpis);
    renderPfTierBar(tierMix);
    renderPfTierMix(tierMix);
    renderPfAppetite(appetite);
    renderPfHardfailReasons(appetite);
    renderPfUwDist(uwDist);
    renderPfQuadrant(quadrant);
    renderPfLobTable(lobTable);
    renderPfGrid(grid);
    renderPfIndustry(industry);
    renderPfTreaties(treaties);
    renderPfBrokerTier(brokerTier);
}

function renderPfKpis(k) {
    const cards = [
        ["Submissions", fmtInt(k.submissions), `${fmtMoney(k.premium_usd, 1)} total premium`],
        ["Total TIV", fmtMoney(k.total_tiv_usd, 1), "all submissions"],
        ["Preferred Premium", fmtMoney(k.preferred_premium_usd, 1), "Preferred tier only"],
        ["Incurred / Loss Ratio", `${fmtMoney(k.incurred_usd, 1)} &middot; ${fmtPct(k.incurred_loss_ratio_pct)}`, "actual, current period"],
        ["Avg UW Score", fmtNum(k.avg_uw_score), "0-100, higher is better"],
        ["Accept Rate", fmtPct(k.accept_rate_pct), "Preferred + Standard"],
        ["Referred to UW", fmtInt(k.referred_count), `${fmtPct(k.referred_pct)} of submissions`],
        ["Hard-Rule Fails", fmtInt(k.hard_rule_fails), `${fmtPct(k.hard_rule_fails_pct)} of submissions`],
    ];
    document.getElementById("pf-kpis").innerHTML = cards.map(([label, value, sub]) =>
        `<div class="kpi-card"><div class="label">${label}</div><div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`
    ).join("");
}

function renderPfTierBar(data) {
    // Segment width is proportional to share, but the text LABEL never lives inside the
    // segment - Preferred regularly sits at ~2% of book, far too narrow to hold its own
    // name+value legibly. A separate legend row below (point 1 below) carries every label
    // at a fixed, always-readable size regardless of how thin its segment is.
    const bar = data.rows.map((r) =>
        `<div style="flex:${Math.max(r.pct_of_book, 0.5)}; background:${TIER_COLORS[r.tier]};" title="${escapeHtml(r.tier)}: ${fmtPct(r.pct_of_book)}"></div>`
    ).join("");
    document.getElementById("pf-tier-bar").innerHTML = `<div style="display:flex; height:30px; border-radius:7px; overflow:hidden; gap:2px;">${bar}</div>`;
    document.getElementById("pf-tier-legend").innerHTML = `<div style="display:flex; flex-wrap:wrap; gap:16px;">
        ${data.rows.map((r) => `<div style="display:flex; align-items:center; gap:6px; font-size:0.84rem;">
            <span style="width:12px; height:12px; border-radius:3px; background:${TIER_COLORS[r.tier]}; flex-shrink:0;"></span>
            <b>${escapeHtml(r.tier)}</b><span class="caption">${fmtPct(r.pct_of_book, 1)}</span>
        </div>`).join("")}
    </div>`;
}

function renderPfTierMix(data) {
    const rows = data.rows.map((r) => `
        <tr><td>${tierChip(r.tier)}</td><td>${fmtInt(r.submissions)}</td><td>${fmtPct(r.pct_of_book)}</td>
        <td>${fmtMoney(r.premium_usd)}</td><td>${fmtNum(r.avg_loss)}</td><td>${fmtNum(r.avg_appetite)}</td>
        <td>${fmtNum(r.avg_bind)}</td><td>${fmtNum(r.avg_uw)}</td></tr>`).join("");
    document.getElementById("pf-tier-mix").innerHTML = `
        <table class="data-table"><thead><tr><th>Tier</th><th>Submissions</th><th>% Book</th><th>Premium</th>
        <th>Avg Loss</th><th>Avg Appetite</th><th>Avg Bind</th><th>Avg UW</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
}

function renderPfAppetite(data) {
    const decisionColorMap = { "Within Appetite": "var(--good)", "Refer – Marginal Fit": "var(--warn)", "Refer – Capacity Constrained": "var(--warn)", "Decline – Outside Appetite": "var(--bad)" };
    const total = data.decisions.reduce((s, d) => s + d.count, 0) || 1;
    const rows = data.decisions.map((d) => {
        const color = decisionColorMap[d.decision] || "var(--text-muted)";
        return `
        <div style="display:flex; align-items:center; gap:8px; font-size:0.84rem; padding:4px 0;">
            <span style="width:190px; color:${color}; font-weight:600;">${escapeHtml(d.decision)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${(d.count / total * 100).toFixed(1)}%; background:${color};"></div></div>
            <span style="width:60px; text-align:right; font-weight:700;">${fmtInt(d.count)}</span>
        </div>`;
    }).join("");
    document.getElementById("pf-appetite").innerHTML = `
        ${rows}
        <div class="caption" style="margin-top:8px;">${fmtInt(data.total_hard_fails)} total hard-rule fails (the 4 real, verified veto flags)</div>`;
}

function renderPfHardfailReasons(data) {
    const rows = data.first_rule_failed.map((r) => `
        <div style="display:flex; align-items:center; gap:8px; font-size:0.84rem; padding:4px 0;">
            <span style="width:170px;">${escapeHtml(r.reason)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${data.total_hard_fails ? (r.count / data.total_hard_fails * 100) : 0}%; background:var(--bad);"></div></div>
            <span style="width:55px; text-align:right; font-weight:700;">${fmtInt(r.count)}</span>
        </div>`).join("");
    document.getElementById("pf-hardfail-reasons").innerHTML = rows;
}

function renderPfUwDist(data) {
    const x = data.bins.map((b) => (b.start + b.end) / 2);
    const y = data.bins.map((b) => b.count);
    const bands = [
        { name: "Decline", range: [0, 38], color: "var(--bad)" }, { name: "High", range: [38, 55], color: "var(--warn)" },
        { name: "Standard", range: [55, 72], color: "var(--accent)" }, { name: "Preferred", range: [72, 100], color: "var(--good)" },
    ];
    const shapes = bands.map((b) => ({
        type: "rect", xref: "x", yref: "paper", x0: b.range[0], x1: b.range[1], y0: 0, y1: 1,
        fillcolor: cssVal(b.color), opacity: 0.18, line: { width: 0 },
    }));
    const annotations = bands.map((b) => ({
        x: (b.range[0] + b.range[1]) / 2, y: 1, yref: "paper", yanchor: "bottom",
        text: b.name, showarrow: false, font: { size: 10, color: cssVal(b.color) },
    }));
    Plotly.newPlot("pf-uw-dist", [{
        x, y, type: "bar", marker: { color: cssVal("var(--navy)") },
        hovertemplate: "UW Score %{x:.0f}<br>%{y:,} submissions<extra></extra>",
    }], {
        shapes, annotations, height: 300, margin: { l: 44, r: 10, t: 24, b: 40 },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: { title: "UW Score (0 = worst, 100 = best)", range: [0, 100] }, yaxis: { title: "Submissions" },
        font: { color: cssVal("var(--text-main)"), size: 11 }, bargap: 0.05,
    }, { displayModeBar: false, responsive: true });
    // Re-filtering calls this again - replace the previous note instead of appending another
    // one each time (a plain .after() on every call would otherwise pile up duplicate
    // captions after a few filter clicks).
    let note = document.getElementById("pf-uw-dist-note");
    if (!note) {
        note = document.createElement("div");
        note.id = "pf-uw-dist-note";
        note.className = "caption";
        note.style.marginTop = "6px";
        document.getElementById("pf-uw-dist").after(note);
    }
    note.textContent = `Loss score sits on its floor (10) for ${fmtInt(data.floor_count)} submissions (${fmtPct(data.floor_pct)}) and on its cap (100) for ${fmtInt(data.cap_count)} (${fmtPct(data.cap_pct)}).`;
}

function renderPfQuadrant(data) {
    const el = document.getElementById("pf-quadrant");
    if (!data.points.length) { el.innerHTML = `<div class="caption">No submissions match the current filter.</div>`; return; }
    const xLo = 0, xHi = 100;
    const yLo = data.y_min, yHi = data.y_max;
    const yPad = Math.max((yHi - yLo) * 0.08, 1);
    const traces = TIER_ORDER.map((tier) => {
        const pts = data.points.filter((p) => p.tier === tier);
        return {
            x: pts.map((p) => p.loss), y: pts.map((p) => p.bind), name: tier, mode: "markers",
            marker: { size: pts.map((p) => Math.max(6, Math.sqrt(p.premium) / 10)), color: cssVal(TIER_COLORS[tier]), opacity: 0.78, line: { width: 0 } },
            text: pts.map((p) => `${p.submission_id}<br>${p.line_of_business}<br>Premium: ${fmtMoney(p.premium)}`),
            hovertemplate: "%{text}<br>Loss %{x:.1f} · Bind %{y:.1f}<extra>" + tier + "</extra>",
        };
    });
    const qColor = { good: cssVal("var(--good)"), warn: cssVal("var(--warn)"), accent: cssVal("var(--accent)"), bad: cssVal("var(--bad)") };
    const alpha = (hex, a) => {
        const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
        return `rgba(${r},${g},${b},${a})`;
    };
    const shapes = [
        { type: "rect", xref: "x", yref: "y", x0: xLo, x1: data.mid_x, y0: data.mid_y, y1: yHi + yPad, fillcolor: alpha(qColor.good, 0.12), line: { width: 0 }, layer: "below" },
        { type: "rect", xref: "x", yref: "y", x0: data.mid_x, x1: xHi, y0: data.mid_y, y1: yHi + yPad, fillcolor: alpha(qColor.warn, 0.12), line: { width: 0 }, layer: "below" },
        { type: "rect", xref: "x", yref: "y", x0: xLo, x1: data.mid_x, y0: yLo - yPad, y1: data.mid_y, fillcolor: alpha(qColor.accent, 0.1), line: { width: 0 }, layer: "below" },
        { type: "rect", xref: "x", yref: "y", x0: data.mid_x, x1: xHi, y0: yLo - yPad, y1: data.mid_y, fillcolor: alpha(qColor.bad, 0.12), line: { width: 0 }, layer: "below" },
        { type: "line", xref: "x", yref: "paper", x0: data.mid_x, x1: data.mid_x, y0: 0, y1: 1, line: { dash: "dot", color: cssVal("var(--navy)"), width: 1.5 } },
        { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: data.mid_y, y1: data.mid_y, line: { dash: "dot", color: cssVal("var(--navy)"), width: 1.5 } },
    ];
    const annotations = [
        { x: xLo + (xHi - xLo) * 0.02, y: yHi + yPad, xanchor: "left", text: "<b>WIN — Preferred</b>", showarrow: false, font: { size: 10, color: qColor.good } },
        { x: xHi - (xHi - xLo) * 0.02, y: yHi + yPad, xanchor: "right", text: "<b>PRICE UP — refer</b>", showarrow: false, font: { size: 10, color: qColor.warn } },
        { x: xLo + (xHi - xLo) * 0.02, y: yLo - yPad, xanchor: "left", yanchor: "bottom", text: "<b>NURTURE — broker push</b>", showarrow: false, font: { size: 10, color: qColor.accent } },
        { x: xHi - (xHi - xLo) * 0.02, y: yLo - yPad, xanchor: "right", yanchor: "bottom", text: "<b>AVOID — decline</b>", showarrow: false, font: { size: 10, color: qColor.bad } },
    ];
    Plotly.newPlot("pf-quadrant", traces, {
        // Plotly axis/annotation text is plain text, not HTML - an "&rarr;" entity renders as
        // the literal 6 characters, not an arrow. Use the real Unicode glyph instead.
        height: 300, margin: { l: 60, r: 10, t: 10, b: 70 }, shapes, annotations,
        xaxis: { title: "Loss Score (worse →)", range: [xLo, xHi], automargin: true },
        yaxis: { title: "Bind Score (more likely →)", range: [yLo - yPad, yHi + yPad], automargin: true },
        legend: { orientation: "h", y: -0.2, font: { size: 10 } },
        paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { color: cssVal("var(--text-main)"), size: 11 },
    }, { displayModeBar: false, responsive: true });
    let note = document.getElementById("pf-quadrant-note");
    if (!note) {
        note = document.createElement("div");
        note.id = "pf-quadrant-note";
        note.className = "caption";
        note.style.marginTop = "6px";
        el.after(note);
    }
    note.textContent = "Bubble size = premium · color = tier · split at this slice's own median on each axis. Click a tier name in the legend to isolate or hide it.";
}

function renderPfLobTable(data) {
    const rows = data.rows.map((r) => {
        const mixBar = TIER_ORDER.map((t) => `<div style="flex:${Math.max(r.tier_mix[t], 0.01)}; background:${TIER_COLORS[t]}; height:14px;" title="${t}: ${r.tier_mix[t]}"></div>`).join("");
        return `<tr><td><b>${escapeHtml(r.lob)}</b></td><td>${fmtInt(r.submissions)}</td><td>${fmtMoney(r.premium_usd)}</td>
            <td style="color:${r.loss_ratio_pct > 80 ? "var(--bad)" : "var(--text-main)"}; font-weight:${r.loss_ratio_pct > 80 ? "700" : "400"};">${fmtPct(r.loss_ratio_pct)}</td>
            <td>${fmtNum(r.avg_uw)}</td><td><div style="display:flex; width:110px; border-radius:4px; overflow:hidden;">${mixBar}</div></td></tr>`;
    }).join("");
    document.getElementById("pf-lob-table").innerHTML = `
        <table class="data-table"><thead><tr><th>LOB</th><th>Submissions</th><th>Premium</th><th>Loss Ratio</th><th>Avg UW</th><th>Tier Mix (P/S/H/D)</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
}

function renderPfGrid(data) {
    const toggle = document.getElementById("pf-grid-metric-toggle");
    const metrics = [["uw_score", "UW Score"], ["loss", "Loss"], ["appetite", "Appetite"], ["bind", "Bind"]];
    toggle.innerHTML = metrics.map(([k, label]) =>
        `<button class="btn ${portfolioState.gridMetric === k ? "active-toggle" : ""}" data-metric="${k}">${label}</button>`).join("");
    toggle.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        portfolioState.gridMetric = b.dataset.metric;
        refreshPortfolio();
    }));

    const vals = data.values.flat().filter((v) => v !== null);
    const vmin = Math.min(...vals), vmax = Math.max(...vals);
    const grade = (v) => {
        if (v === null) return "";
        let t = vmax === vmin ? 0.5 : (v - vmin) / (vmax - vmin);
        if (!data.higher_is_better) t = 1 - t;
        const lerp = (a, b, f) => Math.round(a + (b - a) * f);
        const hex = (c) => c.toString(16).padStart(2, "0");
        const colors = { bad: [192, 57, 43], warn: [183, 149, 11], accent: [59, 110, 199], good: [30, 132, 73] };
        let c1, c2, f;
        if (t < 1 / 3) { c1 = colors.bad; c2 = colors.warn; f = t * 3; }
        else if (t < 2 / 3) { c1 = colors.warn; c2 = colors.accent; f = (t - 1 / 3) * 3; }
        else { c1 = colors.accent; c2 = colors.good; f = (t - 2 / 3) * 3; }
        const r = lerp(c1[0], c2[0], f), g = lerp(c1[1], c2[1], f), bl = lerp(c1[2], c2[2], f);
        return `#${hex(r)}${hex(g)}${hex(bl)}`;
    };
    let html = `<table class="data-table"><thead><tr><th>LOB \\ Region</th>${data.regions.map((r) => `<th>${escapeHtml(r)}</th>`).join("")}</tr></thead><tbody>`;
    data.lobs.forEach((lob, i) => {
        const selected = lob === portfolioState.lob;
        html += `<tr><td style="font-weight:${selected ? "800" : "400"};">${escapeHtml(lob)}</td>`;
        data.regions.forEach((region, j) => {
            const v = data.values[i][j];
            const bg = v === null ? "" : grade(v);
            const isSelected = selected && region === portfolioState.region;
            html += `<td class="grid-cell" data-lob="${escapeHtml(lob)}" data-region="${escapeHtml(region)}" title="${escapeHtml(lob)} &times; ${escapeHtml(region)}: ${v === null ? "no data" : v.toFixed(1)} - click to filter the page"
                style="cursor:pointer; background:${bg}; color:white; font-weight:700; text-align:right;${isSelected ? "outline:3px solid var(--text-main); outline-offset:-3px;" : ""}">${v === null ? "—" : v.toFixed(1)}</td>`;
        });
        html += `</tr>`;
    });
    html += `</tbody></table>`;
    document.getElementById("pf-grid").innerHTML = html;
    document.querySelectorAll(".grid-cell").forEach((cell) => cell.addEventListener("click", () => {
        const isSame = portfolioState.lob === cell.dataset.lob && portfolioState.region === cell.dataset.region;
        portfolioState.lob = isSame ? "All" : cell.dataset.lob;
        portfolioState.region = isSame ? "All" : cell.dataset.region;
        refreshPortfolio();
    }));
}

function renderPfIndustry(data) {
    const x = data.rows.map((r) => r.tiv_share_pct);
    const y = data.rows.map((r) => r.industry);
    const colors = data.rows.map((r) => r.tiv_share_pct > r.cap_pct ? cssVal("var(--bad)") : cssVal("var(--accent)"));
    Plotly.newPlot("pf-industry", [{
        x, y, type: "bar", orientation: "h", marker: { color: colors },
        hovertemplate: "%{y}<br>%{x:.1f}% of book TIV<extra></extra>",
    }], {
        height: 340, margin: { l: 10, r: 40, t: 10, b: 44 }, paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
        shapes: [{ type: "line", x0: 15, x1: 15, y0: -0.5, y1: data.rows.length - 0.5, line: { dash: "dot", color: cssVal("var(--bad)"), width: 1.5 } }],
        annotations: [{ x: 15, y: data.rows.length - 0.5, text: "15% cap", showarrow: false, yanchor: "bottom", font: { size: 10, color: cssVal("var(--bad)") } }],
        // automargin + a couple of spaces of ticksuffix gives the category labels real
        // breathing room instead of running straight into the bars at x=0.
        xaxis: { title: "% of book TIV" }, yaxis: { automargin: true, ticksuffix: "  " },
        font: { color: cssVal("var(--text-main)"), size: 11 },
    }, { displayModeBar: false, responsive: true });
}

function renderPfTreaties(data) {
    const rows = data.rows.map((r) => `<tr><td>${escapeHtml(r.level)}</td>
        <td style="color:${r.min_headroom_pct < 0 ? "var(--bad)" : "var(--good)"}; font-weight:700;">${fmtPct(r.min_headroom_pct)}</td>
        <td>${fmtPct(r.avg_headroom_pct)}</td></tr>`).join("");
    document.getElementById("pf-treaties").innerHTML = `
        <table class="data-table"><thead><tr><th>Level</th><th>Min Headroom</th><th>Avg Headroom</th></tr></thead>
        <tbody>${rows}</tbody></table>
        <div class="caption" style="margin-top:8px;">Aggregated across the current filter - not broken out by zone/peril yet.</div>`;
}

function renderPfBrokerTier(data) {
    const rows = data.rows.map((r) => `<tr><td><b>${escapeHtml(r.broker_tier)}</b></td><td>${fmtInt(r.submissions)}</td>
        <td>${fmtNum(r.avg_bind)}</td><td>${fmtNum(r.avg_uw)}</td></tr>`).join("");
    document.getElementById("pf-broker-tier").innerHTML = `
        <table class="data-table"><thead><tr><th>Tier</th><th>Submissions</th><th>Avg Bind</th><th>Avg UW</th></tr></thead>
        <tbody>${rows}</tbody></table>`;
}
