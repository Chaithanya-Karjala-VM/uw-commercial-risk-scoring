"""
Deterministic narrative generation for the What-If Simulator's result summary. Not an LLM
call - every sentence is assembled from real computed numbers using pre-written sentence
patterns, chosen by a stable hash of the submission's own scores. Same submission always
reads the same way (genuinely reproducible, matching the "Technical & audit detail" panel's
own promise); different submissions land on different phrasing so it doesn't read as one
template copy-pasted with numbers swapped in.

Plain prose only - no markdown bold syntax (**text**) inside the body text. Bold rendering
inside a raw HTML block isn't reliable in Streamlit, and literal asterisk characters leaking
into a client-facing page read as broken, not as emphasis - found this exact problem when
first shipping this panel. Emphasis is handled by the page's own HTML/CSS instead, not by
markdown characters embedded in the text itself.
"""
import hashlib


def _pick(variants, *seed_parts):
    """Stable variant selection - same seed always picks the same variant, different
    submissions usually land on different ones. Not Python's random (which reseeds per
    process) - this is deterministic across restarts too, which matters for a panel that
    explicitly claims reproducibility."""
    key = "|".join(str(p) for p in seed_parts)
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return variants[h % len(variants)]


def _lbl(d):
    return d["label"].lower() if d else ""


def build_banner_summary(result, loss_drv, bind_drv):
    """One clean sentence for the result banner, naming the actual top drivers - not the
    short generic reason text (e.g. "Strong loss profile" alone), which reads as too thin
    for a client-facing headline."""
    tier = result["risk_tier_v2"]
    if result["appetite_capacity_score_v2"] == 0:
        return "Fails the compliance/appetite hard-rule gate - declined regardless of the loss and bind view."

    clauses = []
    if loss_drv:
        d = loss_drv[0]
        verb = "raises" if d["direction"] == "increases" else "lowers"
        clauses.append(f"{_lbl(d)}, which {verb} the expected loss cost")
    if bind_drv:
        d = bind_drv[0]
        verb = "improves" if d["direction"] == "increases" else "reduces"
        clauses.append(f"{_lbl(d)}, which {verb} the likelihood to bind")

    if not clauses:
        return f"{tier} band - no single field dominates this read; a balanced profile overall."
    return f"{tier} band, driven mainly by " + "; ".join(clauses) + "."


def build_narrative(result, loss_drv, bind_drv, appetite_drv):
    tier = result["risk_tier_v2"]
    uw = result["uw_score_v2"]
    loss = result["loss_propensity_score_v2"]
    bind = result["bind_propensity_score_v2"]
    appetite = result["appetite_capacity_score_v2"]
    seed = (tier, round(uw), round(loss), round(bind), round(appetite))

    sentences = []

    # --- opening: tier + composite score ---
    opening = _pick([
        f"This submission lands in the {tier} band with a composite underwriting score of {uw:.1f} out of 100.",
        f"Overall, the model places this risk in the {tier} category, scoring {uw:.1f} out of 100 on the combined view.",
        f"The blended underwriting view here is {tier} - a composite score of {uw:.1f} out of 100 across loss, bind, and appetite.",
        f"Taking loss, bind, and appetite together, this comes out {tier}, at {uw:.1f} out of 100 overall.",
    ], "open", *seed)
    sentences.append(opening)

    # --- loss propensity ---
    if loss <= 25:
        loss_desc = _pick(["a strong, low-risk loss profile", "a genuinely clean loss outlook",
                            "loss expectations well below the concerning range"], "loss_good", *seed)
    elif loss <= 55:
        loss_desc = _pick(["a moderate loss outlook with some watch items", "a middling loss picture, not alarming but not clean either",
                            "a loss view that sits in the middle of the range"], "loss_mid", *seed)
    else:
        loss_desc = _pick(["an elevated loss outlook that needs attention", "a loss profile on the wrong side of acceptable",
                            "a genuinely concerning loss picture"], "loss_bad", *seed)
    if loss_drv:
        top = _lbl(loss_drv[0])
        loss_sentence = _pick([
            f"Loss Propensity scores {loss:.1f} out of 100 - {loss_desc}, with {top} the single biggest factor pulling it in that direction.",
            f"On the loss side ({loss:.1f}/100), this reflects {loss_desc}; {top} is doing most of the work here.",
            f"The Loss Propensity Score of {loss:.1f} points to {loss_desc}, driven mainly by {top}.",
        ], "loss_sent", *seed)
    else:
        loss_sentence = f"Loss Propensity scores {loss:.1f} out of 100 - {loss_desc}, without any single field dominating the read."
    sentences.append(loss_sentence)

    # --- bind propensity ---
    if bind >= 55:
        bind_desc = _pick(["a good chance of closing", "a healthy likelihood of binding", "solid bind prospects"], "bind_good", *seed)
    elif bind >= 35:
        bind_desc = _pick(["middling bind odds", "an average likelihood of closing", "a fairly typical shot at binding"], "bind_mid", *seed)
    else:
        bind_desc = _pick(["a real risk of not closing", "weak bind odds", "a low likelihood of actually binding"], "bind_bad", *seed)
    if bind_drv:
        top = _lbl(bind_drv[0])
        bind_sentence = _pick([
            f"Bind Propensity sits at {bind:.1f} out of 100 - {bind_desc}, most influenced by {top}.",
            f"On bind likelihood ({bind:.1f}/100), this points to {bind_desc}; {top} stands out as the biggest lever.",
            f"With a Bind Propensity of {bind:.1f}, expect {bind_desc} - {top} is the clearest driver.",
        ], "bind_sent", *seed)
    else:
        bind_sentence = f"Bind Propensity sits at {bind:.1f} out of 100 - {bind_desc}."
    sentences.append(bind_sentence)

    # --- appetite ---
    if appetite == 0:
        appetite_sentence = _pick([
            "The submission fails the hard-rule appetite gate outright - this is a decline regardless of the loss and bind view above.",
            "Appetite/Capacity is hard-gated to zero here: a compliance or restricted-operations flag forces a decline no matter what the other scores say.",
        ], "appetite_fail", *seed)
    elif appetite >= 60:
        appetite_sentence = _pick([
            f"Appetite/Capacity is strong at {appetite:.1f} out of 100 - this is squarely within where the book wants to grow.",
            f"At {appetite:.1f}/100, Appetite/Capacity is comfortably positive - good fit, real capacity headroom.",
        ], "appetite_good", *seed)
    else:
        appetite_sentence = _pick([
            f"Appetite/Capacity comes in at {appetite:.1f} out of 100 - passes the hard-rule gate, but fit or capacity headroom is only moderate.",
            f"At {appetite:.1f}/100, Appetite/Capacity clears the gate without being a clean, obvious fit.",
        ], "appetite_mid", *seed)
    sentences.append(appetite_sentence)

    # --- premium ---
    loading = result.get("premium_loading_pct")
    technical = result["indicated_annual_premium_usd"]
    quoted = result.get("quoted_premium_usd")
    if loading is not None and loading == loading and loading > 0:  # not NaN, not zero
        sentences.append(_pick([
            f"The burning cost of ${technical:,.0f} carries a {loading*100:.0f}% rate load for this tier, quoting at ${quoted:,.0f}.",
            f"Because of the {tier} tier, the ${technical:,.0f} burning cost is loaded {loading*100:.0f}% to a ${quoted:,.0f} quote.",
        ], "premium", *seed))
    elif quoted != quoted:  # NaN - Decline tier, not quoted
        sentences.append(_pick([
            f"At a burning cost of ${technical:,.0f}, this isn't quoted at this tier - it needs to be referred first.",
            f"The ${technical:,.0f} burning cost is informational only here; Decline-tier submissions aren't auto-quoted.",
        ], "premium_decline", *seed))

    # deliberately no closing "recommended action" sentence here - that's already the
    # headline of the result banner above this panel, restating it again read as redundant
    return " ".join(sentences)
