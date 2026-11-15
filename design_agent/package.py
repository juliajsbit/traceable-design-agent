"""Design Package: one document a reviewer can read, print and sign, built from a recorded run.

It is built only from the recording, so it needs no model call and says nothing the run did not
record. Every table names the step that produced it, and the evidence ledger at the end lists
every decision in order with its hash (ledger.py).

    release status -> requirements and their sources -> verification matrix -> alternatives
    -> red team -> fragility -> chief engineer review -> compliance -> procurement
    -> open items -> change history -> agents and cost -> evidence ledger
"""
from html import escape as esc

from . import ledger

CSS = """
:root { --text:#16181d; --muted:#5f6672; --line:#d9dce3; --soft:#f4f5f8; --ok:#1f7a4d; --ok-bg:#e6f4ec;
        --bad:#b4372f; --bad-bg:#fbe9e7; --warn:#8a5a00; --warn-bg:#fdf3dc; }
* { box-sizing:border-box; }
body { margin:0; background:#e9ebf0; color:var(--text); font:14px/1.5 -apple-system, "Segoe UI", system-ui, sans-serif; }
main { max-width:980px; margin:24px auto; background:#fff; padding:44px 52px 56px; box-shadow:0 1px 8px rgba(0,0,0,.08); }
h1 { font-size:26px; margin:0 0 2px; letter-spacing:-.01em; }
h2 { font-size:16px; margin:34px 0 4px; padding-top:14px; border-top:2px solid var(--text); }
h2 span { color:var(--muted); font-weight:500; margin-right:8px; }
.by { color:var(--muted); font-size:12.5px; margin:0 0 10px; }
.kicker { font-size:11px; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.muted { color:var(--muted); } .small { font-size:12.5px; }
.status { display:flex; gap:18px; align-items:center; margin:18px 0 6px; padding:14px 18px; border-radius:8px; border:1px solid var(--line); }
.status.ship { background:var(--ok-bg); border-color:var(--ok); } .status.held { background:var(--bad-bg); border-color:var(--bad); }
.status b { font-size:19px; } .status .big { font-size:13px; font-weight:800; letter-spacing:.12em; }
.status.ship .big { color:var(--ok); } .status.held .big { color:var(--bad); }
.facts { display:grid; grid-template-columns:repeat(4, 1fr); gap:10px; margin:12px 0; }
.facts div { background:var(--soft); border-radius:6px; padding:8px 10px; } .facts span { display:block; font-size:11px; color:var(--muted); }
.facts b { font-size:13px; word-break:break-all; }
table { border-collapse:collapse; width:100%; font-size:12.5px; font-variant-numeric:tabular-nums; margin:6px 0 4px; }
th, td { padding:5px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { color:var(--muted); font-weight:600; font-size:11.5px; } tr.pick td { background:var(--ok-bg); }
.pill { display:inline-block; padding:0 7px; border-radius:99px; font-size:11.5px; font-weight:700; white-space:nowrap; }
.ok { color:var(--ok); background:var(--ok-bg); } .bad { color:var(--bad); background:var(--bad-bg); }
.warn { color:var(--warn); background:var(--warn-bg); } .info { color:var(--muted); background:var(--soft); }
.note { background:var(--soft); border-left:3px solid var(--line); padding:8px 12px; margin:8px 0; font-size:13px; }
.note.warnbox { border-color:var(--warn); background:var(--warn-bg); }
ul { margin:6px 0; padding-left:20px; } li { margin:2px 0; }
code { font:12px ui-monospace, Menlo, monospace; }
.sign { display:grid; grid-template-columns:repeat(3, 1fr); gap:26px; margin-top:26px; }
.sign div { border-top:1px solid var(--text); padding-top:4px; font-size:12px; color:var(--muted); height:54px; }
@media print { body { background:#fff; } main { margin:0; box-shadow:none; padding:0; max-width:none; }
  h2 { break-after:avoid; } tr { break-inside:avoid; } a { color:inherit; text-decoration:none; } }
"""

LABEL = {"verified": "ok", "fails": "bad", "fragile": "warn", "not manufacturable": "warn", "out of envelope": "bad"}
STATUS = {"pass": "ok", "fail": "bad", "action": "warn", "confirm": "warn", "info": "info"}


def pill(text, cls="info") -> str:
    return f'<span class="pill {cls}">{esc(str(text))}</span>'


def table(head: list[str], rows: list[list], pick: int | None = None) -> str:
    """Cells are escaped unless they are already HTML (a tuple ("html", text))."""
    def cell(c):
        return c[1] if isinstance(c, tuple) else esc("-" if c is None else str(c))
    body = "".join(f'<tr{" class=pick" if i == pick else ""}>' + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>"
                   for i, r in enumerate(rows))
    return "<table><tr>" + "".join(f"<th>{esc(h)}</th>" for h in head) + "</tr>" + body + "</table>"


def html(s: str):
    return ("html", s)


def design_text(d: dict | None) -> str:
    return f"{d['material']}, {d['width_mm']:g} x {d['thickness_mm']:g} mm" if d else "no design"


def section(n: int, title: str, by: str, body: str) -> str:
    return f"<h2><span>{n}</span>{esc(title)}</h2><p class=by>{esc(by)}</p>{body}"


def build(events: list[dict], name: str = "") -> str:
    def last(kind):
        return next((e for e in reversed(events) if e["type"] == kind), None)

    def every(kind):
        return [e for e in events if e["type"] == kind]

    start, decision, review, end = last("run_start") or {}, last("decision"), last("review"), last("run_end")
    selected = decision["selected"] if decision and decision.get("selected") else None
    req = (last("approved") or last("design_start") or last("requirements") or start).get("requirement") or {}
    chain = ledger.verify(events)
    open_items: list[tuple[str, str]] = []   # (source, text), collected from every section

    # ---- cover ----
    if not selected:
        cls, word, line = "held", "NO DESIGN", (decision or {}).get("reason", "the run did not reach a decision")
    elif review and review["decision"] == "escalate":
        cls, word, line = "held", "HELD", "The review did not release this design. A person has to decide."
    elif review:
        cls, word, line = "ship", "RELEASED", f"Released by the Chief Engineer review, round {review['round']} of {review['max_rounds']}."
    else:
        cls, word, line = "ship", "SELECTED", "Selected by the Verifier. This run has no Chief Engineer review."
    integrity = ("verified: every event matches its hash" if chain["ok"] else
                 f"BROKEN at event {chain['broken_at']}: the recording was altered" if chain["chained"] else
                 "not chained: recorded before the ledger existed")
    out = [f"<div class=kicker>Design Package</div><h1>{esc(req.get('part', 'Bracket'))}</h1>",
           f"<div class=muted>{esc(name)}</div>",
           f"<div class='status {cls}'><span class=big>{word}</span><div><b>{esc(design_text(selected['design']) if selected else 'No design')}"
           f"{f', {selected['mass_g']:.0f} g' if selected else ''}</b><div class=small>{esc(line)}</div></div></div>",
           "<div class=facts>" + "".join(f"<div><span>{esc(k)}</span><b>{esc(str(v))}</b></div>" for k, v in [
               ("model", start.get("model", "-")), ("code version", start.get("code_version", "not recorded")),
               ("run time", f"{end['seconds']} s" if end else "did not finish"), ("events", f"{chain['events']}"),
               ("ledger head", chain["head"] or "-"), ("ledger integrity", integrity),
               ("physics checks", len(every("check"))), ("proposals", len(every("verified")))]) + "</div>",
           "<div class='note warnbox small'>Toy model: textbook cantilever formulas and typical handbook values. The package shows "
           "the pattern of a traceable design record. It is not an engineering release and not a compliance determination.</div>"]
    n = 0

    def add(title, by, body):
        nonlocal n
        n += 1
        out.append(section(n, title, by, body))

    # ---- requirements ----
    desk, approved = last("requirements"), last("approved")
    rows = [["load at the tip", f"{req.get('load_n', 0):.0f} N"], ["arm length", f"{req.get('length_mm')} mm"],
            ["tip deflection limit", f"{req.get('max_deflection_mm')} mm"],
            ["factors of safety", f"{req.get('yield_factor')} on yield, {req.get('ultimate_factor')} on ultimate"],
            ["width", " to ".join(f"{v:g}" for v in req.get("width_range_mm", [])) + " mm"],
            ["plate thickness", " to ".join(f"{v:g}" for v in req.get("thickness_range_mm", [])) + " mm"],
            ["robustness case", f"+{req.get('load_uncertainty', 0):.0%} load, {req.get('thickness_tolerance_mm')} mm thinner plate"]]
    body = table(["requirement", "value"], rows)
    if desk and desk.get("mode") == "brief":
        body = f"<div class=note><b>Mission brief.</b> {esc(desk['brief'])}</div>" + body
        body += "<p class=small><b>Where each value came from.</b></p>" + table(
            ["field", "value", "source", "quote or reason"],
            [[s["field"], s["value"], html(pill(s["source"], "ok" if s["source"] == "brief" else "warn")), s["note"]] for s in desk["draft"]["sources"]])
        body += "".join(f"<p class=small>Computed by code: {esc(d)}</p>" for d in desk["derived"])
        body += "".join(f"<p class=small>Assumed: {esc(a)}</p>" for a in desk["assumed"])
        open_items += [("Requirements Desk", q) for q in desk["draft"]["open_questions"]]
    if approved:
        edits = ", ".join(f"{k} = {v}" for k, v in (approved.get("edits") or {}).items())
        body += f"<div class=note>Approved by a person{' with edits: ' + esc(edits) if edits else ' as drafted'}, at {approved.get('t')} s.</div>"
    add("Requirements", "Requirements Desk: an agent drafts, code computes, a person approves", body)

    # ---- verification matrix ----
    if selected:
        ref = f"event {decision['seq']}" if decision.get("seq") else "the decision event"
        add("Verification matrix", f"Verifier: plain code, no model. Evidence: {ref}",
            table(["id", "requirement", "result", "limit", "method", "status"],
                  [[r["id"], r["title"], r["value"], r["limit"], r["method"], html(pill("pass" if r["passed"] else "fail", "ok" if r["passed"] else "bad"))]
                   for r in selected["matrix"]]) + f"<p class=small>Why this design: {esc(decision['reason'])}.</p>")

    # ---- alternatives ----
    props = {e["id"]: e for e in every("verified")}   # after a change run the same id is checked again; the last check counts
    if props:
        ordered = sorted(props.values(), key=lambda e: (e["label"] != "verified", e["mass_g"] or 1e9))
        false_pass = sum(e["false_pass"] for e in ordered)
        rows = [[e["id"], e["agent"], f"{e['design']['width_mm']:g} x {e['design']['thickness_mm']:g} mm",
                 f"{e['mass_g']:.0f} g" if e["mass_g"] else "-", html(pill(e["label"], LABEL.get(e["label"], "info"))),
                 ("FALSE PASS. " if e["false_pass"] else "") + ("; ".join(e["findings"]) or (e["reason"] if e["label"] != "verified" else ""))]
                for e in ordered]
        pick = next((i for i, e in enumerate(ordered) if selected and e["id"] == selected["id"]), None)
        add("Alternatives considered", "Specialist agents propose and claim; the Verifier recomputes every claim",
            f"<p class=small>{len(ordered)} proposals, {sum(e['label'] == 'verified' for e in ordered)} verified, "
            f"{false_pass} false passes caught (an agent said it passes, the physics says it does not).</p>"
            + table(["id", "proposed by", "section", "mass", "verdict", "what the Verifier found"], rows, pick))

    # ---- red team ----
    red = last("redteam")
    if red:
        bp = red["breaking_point"]
        rows = []
        for a in red["attacks"]:
            cls = {"confirmed": "bad", "refuted": "ok", "needs analysis": "warn"}[a["result"]]
            wrong = " The agent predicted the opposite." if a.get("agent_right") is False else ""
            rows.append([a["title"], a["rationale"], a["severity"], html(pill(a["result"], cls)),
                         (f"MS {a['margin_yield']:+.2f}, {a['deflection_mm']:.2f} mm." if "margin_yield" in a else "No tool here can test it.") + wrong])
            if a["result"] == "needs analysis":
                open_items.append(("Red Team", f"{a['title']}: {a['rationale']}"))
        add("Red Team Review", "An agent attacks the selected design; code re-runs every attack",
            f"<p class=small>Breaking point, computed by code: survives up to x{bp['load_multiplier']} load and a plate "
            f"{bp['thickness_loss_mm']} mm thinner.</p>" + table(["attack", "why", "severity", "code result", "numbers"], rows))

    # ---- fragility ----
    frag, trade = last("fragility"), last("tradeoffs")
    if frag:
        sens = ", ".join(f"{k} {v:+.2f}" for k, v in sorted(frag["sensitivity"].items(), key=lambda kv: -abs(kv[1])))
        body = (f"<p>{frag['failures']} of {frag['samples']} sampled parts fail: <b>P(failure) {frag['p_failure']:.2%}</b>. "
                f"5% of parts have a governing margin below {frag['p5']:+.3f}.</p><p class=small>Correlation of each input with the margin: {esc(sens)}.</p>")
        if trade:
            body += "<p class=small><b>Trade-offs across verified designs.</b></p>" + table(
                ["id", "design", "mass", "cost each", "worst-case margin", "P(failure)"],
                [[r["id"], design_text(r["design"]), f"{r['mass_g']:.0f} g", f"${r['cost_usd']:.2f}", f"{r['margin_worst']:+.2f}", f"{r['p_failure']:.2%}"]
                 for r in trade["rows"]], next((i for i, r in enumerate(trade["rows"]) if r["id"] == trade.get("selected_id")), None))
        add("Fragility", "Monte Carlo in plain code, the same seed every run", body)

    # ---- chief engineer ----
    reviews = every("review")
    if reviews:
        body = ""
        for r in reviews:
            cls = {"ship": "ok", "redesign": "warn", "escalate": "bad"}[r["decision"]]
            body += f"<p>{pill(r['decision'].upper(), cls)} <b>Round {r['round']} of {r['max_rounds']}</b>, {esc(design_text(r['design']))}. {esc(r.get('rationale', ''))}</p>"
            if r.get("overruled"):
                body += f"<div class='note warnbox'>The agent proposed {esc(str(r['proposed']))}. Code overruled it: {esc(r['rule'])}.</div>"
            body += table(["id", "source", "finding", "value", "limit", ""],
                          [[f["id"], f["source"], f["title"], f["value"], f["limit"], html(pill("blocking", "bad") if f["blocking"] else pill("ok", "ok"))]
                           for f in r["findings"]])
            if r.get("directive"):
                body += f"<div class=note><b>Directive to the team.</b> {esc(r['directive'])}</div>"
            open_items += [("Chief Engineer", c) for c in r.get("conditions", [])]
            if r["decision"] == "escalate":
                open_items.append(("Chief Engineer", "Not released. Decide: accept the risk with a waiver, relax a requirement, or open the envelope."))
        add("Chief Engineer review", "Code marks the blocking findings; an agent proposes; code has the last word", body)

    # ---- compliance ----
    comp, creview = last("compliance"), last("compliance_review")
    if comp:
        count = lambda k: sum(i["status"] == k for i in comp["items"])   # counted here: early recordings carry no summary
        body = (f"<p>{count('pass')} pass, {count('action') + count('confirm')} need evidence or action, {count('fail')} fail, "
                f"{count('info')} for information.</p>") + table(
            ["id", "area", "rule", "standard", "found", "limit", "status"],
            [[i["id"], i.get("category"), i["title"] + (". " + i["detail"] if i.get("detail") else ""), i.get("standard"), i.get("value"),
              i.get("limit"), html(pill(i["status"], STATUS.get(i["status"], "info")))] for i in comp["items"]])
        if creview:
            body += f"<div class=note><b>Review agent.</b> {esc(creview['summary'])}</div>"
            open_items += [(f"Compliance {a['rule_id']}, owner {a['owner']}", a["action"]) for a in creview["actions"]]
        else:
            open_items += [(f"Compliance {i['id']}", i["title"] + ": " + i.get("detail", "")) for i in comp["items"] if i["status"] in ("action", "fail")]
        add("Compliance", "Compliance Engine: every rule checked in code. A review agent explains and changes no status", body)

    # ---- procurement ----
    rfq, quotes, proc, market = last("rfq"), {q["vendor"]: q for q in every("quote")}, last("procurement"), last("market")
    if rfq or market:
        body = f"<div class=note>{esc(rfq['text'])}</div>" if rfq else "<p>No request for quote went out: the design was not released.</p>"
        if quotes:
            rank = {r["vendor"]: i for i, r in enumerate(proc["ranking"], 1)} if proc else {}
            rows = []
            for q in sorted(quotes.values(), key=lambda q: rank.get(q["vendor"], 99)):
                bid = q["quote"]
                checks = "; ".join(q["flags"] + q["blockers"]) or "eligible, price matches the shop estimate"
                rows.append([rank.get(q["vendor"], "-"), q["name"], q["location"],
                             f"${bid['price_each_usd']:.2f}" if bid["bid"] and bid["price_each_usd"] else "no bid",
                             f"{bid['lead_time_days']} days" if bid["bid"] and bid["lead_time_days"] else "-", checks])
            body += table(["rank", "vendor", "location", "price each", "lead time", "code checks"], rows, 0 if rank else None)
        if proc and proc.get("memo"):
            body += f"<div class=note><b>Procurement memo.</b> {esc(proc['memo'])}</div>"
        if market and market["vendors"]:
            def link(v):
                url = v.get("url") or ""
                return html(f'<a href="{esc(url, quote=True)}">{esc(v["name"])}</a>') if url.startswith(("http://", "https://")) else v["name"]
            body += "<p class=small><b>Market scan.</b> Real shops outside the approved list, kept only if their link was in the search results.</p>" + table(
                ["shop", "location", "capabilities", "certifications"],
                [[link(v), v.get("location"), v.get("capabilities"), v.get("certifications")] for v in market["vendors"]])
        add("Procurement", "Vendor agents quote; code checks eligibility and price, and ranks; an agent writes the memo. Sample vendors", body)

    # ---- open items ----
    if open_items:
        add("Open items", "Everything this run could not close, in one list",
            table(["#", "raised by", "item"], [[i, src, text] for i, (src, text) in enumerate(open_items, 1)]))

    # ---- change history ----
    impact = last("impact")
    if impact:
        steps = every("step")
        body = "<p>" + (" ".join(pill(f"{c['field']}: {c['old']:g} -> {c['new']:g} {c['unit']} ({c['effect']})", "warn") for c in impact["changes"])
                        or pill("nothing changed")) + f"</p><p>Design search {esc(impact['search'])}: {esc(impact['why'])}. "
        body += f"{impact['still_verified']} of {impact['were_verified']} old designs are still verified.</p>"
        body += table(["step", "", "why"], [[s["step"], html(pill("reused", "ok") if s["action"] == "reused" else pill("ran again", "warn")),
                                             "inputs unchanged" if s["action"] == "reused" else "changed: " + ", ".join(s.get("changed", []))] for s in steps])
        add("Change history", "This run started from an earlier run with a changed requirement", body)

    # ---- agents and cost ----
    usage = {e["agent"]: e for e in every("usage")}
    if usage:
        total = sum(u.get("cost_usd") or 0 for u in usage.values())
        add("Agents and cost", "Model calls made in this run. Reused steps cost nothing and are not listed",
            table(["agent", "model calls", "tool calls", "tokens in", "tokens out", "cost"],
                  [[a, u["llm_calls"], u.get("tool_calls", 0), u["input_tokens"], u["output_tokens"], f"${u.get('cost_usd') or 0:.4f}"]
                   for a, u in usage.items()] + [["total", "", "", "", "", f"${total:.4f}"]]))

    # ---- ledger ----
    rows = [[r["seq"] or "-", html(f"<span style='white-space:nowrap'>{r['t']} s</span>") if r["t"] is not None else "-", r["who"], r["kind"],
             r["what"] + (" (reused from the baseline)" if r["reused"] else ""), html(f"<code>{esc(r['hash'] or '-')}</code>")]
            for r in ledger.entries(events)]
    add("Evidence ledger", "Every decision in order. Each event is hashed together with the one before it",
        f"<p class=small>Integrity: {esc(integrity)}. Head: <code>{esc(chain['head'] or '-')}</code>. The full event stream, "
        f"{chain['events']} events, is the recording <code>{esc(name)}.jsonl</code>.</p>"
        + table(["event", "time", "who", "kind", "what", "hash"], rows))

    out.append("<div class=sign><div>Design lead</div><div>Chief engineer</div><div>Quality</div></div>")
    title = f"Design Package - {name}" if name else "Design Package"
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{esc(title)}</title><style>{CSS}</style></head><body><main>{''.join(out)}</main></body></html>")
