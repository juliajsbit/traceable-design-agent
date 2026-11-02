"""HTML reports: one decision trace, and the eval scoreboard across proposers and cases."""
from html import escape

from .evals import CaseResult, scoreboard
from .pipeline import summarize
from .schemas import DecisionTrace, Record

CSS = """
:root { --bg:#f7f7f5; --card:#fff; --text:#1c1c1a; --muted:#6b6b66; --line:#e3e3de;
        --ok:#1f7a4d; --ok-bg:#e6f4ec; --bad:#b4372f; --bad-bg:#fbe9e7; --warn:#8a5a00; --warn-bg:#fdf3dc; }
@media (prefers-color-scheme: dark) { :root { --bg:#151514; --card:#1f1f1d; --text:#ececea; --muted:#9a9a94;
        --line:#33332f; --ok:#5cc791; --ok-bg:#16301f; --bad:#f08a80; --bad-bg:#3a1c19; --warn:#e8b54f; --warn-bg:#352a12; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 -apple-system, system-ui, sans-serif; }
main { max-width:1180px; margin:0 auto; padding:32px 16px 64px; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:17px; margin:32px 0 12px; }
.muted { color:var(--muted); }
.tiles { display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:12px; margin-top:20px; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }
.tile b { display:block; font-size:24px; font-variant-numeric:tabular-nums; }
.tile.bad b { color:var(--bad); } .tile.ok b { color:var(--ok); } .tile.warn b { color:var(--warn); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:18px; }
.selected { border-left:4px solid var(--ok); }
.scroll { overflow-x:auto; background:var(--card); border:1px solid var(--line); border-radius:10px; }
table { border-collapse:collapse; width:100%; font-size:13px; font-variant-numeric:tabular-nums; }
th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; white-space:nowrap; }
th { color:var(--muted); font-weight:600; } td.wrap { white-space:normal; min-width:280px; }
tr.false-pass { background:var(--bad-bg); } tr.is-selected { background:var(--ok-bg); }
.pill { display:inline-block; padding:1px 8px; border-radius:99px; font-size:12px; font-weight:600; }
.l-verified, .pass { color:var(--ok); background:var(--ok-bg); }
.l-fails, .fail { color:var(--bad); background:var(--bad-bg); }
.l-fragile, .l-not-manufacturable, .l-out-of-envelope { color:var(--warn); background:var(--warn-bg); }
.f-false_pass { color:var(--bad); font-weight:600; } .f-claim_off, .f-out_of_bounds { color:var(--warn); }
.matrix td:first-child { font-weight:600; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{escape(title)}</title><style>{CSS}</style></head>
<body><main>{body}
<p class="muted" style="margin-top:28px">Toy model: textbook cantilever formulas, typical handbook material values, illustrative factors of safety
and stock sizes. It shows the pattern (propose, verify against physics, trace every decision to a requirement), not real engineering.</p>
</main></body></html>
"""


def _num(x, fmt=".2f") -> str:
    return "-" if x is None else format(x, fmt)


def _pill(label: str) -> str:
    return f'<span class="pill l-{label.replace(" ", "-")}">{escape(label)}</span>'


def _row(r: Record, selected_id: str | None) -> str:
    p, c, d = r.proposal, r.check, r.proposal.design
    cls = "is-selected" if p.id == selected_id else ("false-pass" if any(f.kind == "false_pass" for f in r.findings) else "")
    findings = "<br>".join(f'<span class="f-{f.kind}">{escape(f.kind)}</span>: {escape(f.message)}' for f in r.findings)
    return f"""<tr class="{cls}">
<td>{escape(p.id)}</td><td>{escape(d.material)}</td><td>{d.width_mm:g} x {d.thickness_mm:g}</td>
<td>{_num(p.claimed_margin_yield, "+.2f")} / {_num(c and c.nominal.margin_yield, "+.2f")}</td>
<td>{_num(c and c.worst_case.margin_yield, "+.2f")}</td>
<td>{_num(p.claimed_deflection_mm)} / {_num(c and c.nominal.deflection_mm)}</td>
<td>{_num(p.claimed_mass_g, ".0f")} / {_num(c and c.mass_g, ".0f")}</td>
<td>{_pill(r.label)}</td>
<td class="wrap">{escape(r.reason)}{"<br>" + findings if findings else ""}</td></tr>"""


def _matrix(r: Record) -> str:
    rows = "\n".join(
        f'<tr><td>{x.id}</td><td>{escape(x.title)}</td><td>{x.method}</td><td>{escape(x.value)}</td>'
        f'<td>{escape(x.limit)}</td><td><span class="pill {"pass" if x.passed else "fail"}">{"pass" if x.passed else "fail"}</span></td></tr>'
        for x in r.check.results)
    return f"""<div class="scroll"><table class="matrix"><thead><tr><th>requirement</th><th>title</th><th>method</th>
<th>result</th><th>limit</th><th>status</th></tr></thead><tbody>{rows}</tbody></table></div>"""


def render(trace: DecisionTrace) -> str:
    s = summarize(trace)
    req = trace.requirement
    sel = next((r for r in trace.records if r.proposal.id == trace.selected_id), None)
    who = escape(trace.proposer + (f" ({trace.model})" if trace.model else ""))
    u = trace.usage
    usage = (f" {u.llm_calls} model calls, {u.tool_calls} tool calls, {u.input_tokens + u.output_tokens:,} tokens"
             + (f", ${u.cost_usd:.2f}" if u.cost_usd else "") + ".") if u.llm_calls else ""

    if sel:
        d, c = sel.proposal.design, sel.check
        selected = f"""<div class="card selected"><b>{escape(sel.proposal.id)}: {escape(d.material)}, {d.width_mm:g} x {d.thickness_mm:g} mm, {c.mass_g:.0f} g</b>
<p>{escape(trace.selection_reason)}.</p>
<p class="muted">Root bending stress {c.nominal.stress_mpa:.0f} MPa at {req.load_n:g} N, {c.worst_case.stress_mpa:.0f} MPa in the worst case
({c.worst_case.load_n:g} N, {c.worst_case.thickness_mm:g} mm plate). Every value below is computed, not taken from the proposer.
Proposer's rationale: "{escape(sel.proposal.rationale)}"</p></div>
<h2>Verification matrix for {escape(sel.proposal.id)}</h2>{_matrix(sel)}"""
    else:
        selected = f'<div class="card">{escape(trace.selection_reason)}</div>'

    rows = "\n".join(_row(r, trace.selected_id) for r in trace.records)
    body = f"""<h1>Design decision trace, case {escape(req.case)}</h1>
<p class="muted">{escape(req.part)}: {req.load_n:g} N at {req.length_mm:g} mm. Proposer: {who}, {trace.rounds} round(s),
{trace.seconds:g} s.{usage} {escape(trace.created_at)}</p>
<div class="tiles">
<div class="tile"><span class="muted">proposals</span><b>{s['proposals']}</b></div>
<div class="tile ok"><span class="muted">verified</span><b>{s['verified']}</b></div>
<div class="tile"><span class="muted">fail physics</span><b>{s['fails']}</b></div>
<div class="tile warn"><span class="muted">fragile</span><b>{s['fragile']}</b></div>
<div class="tile warn"><span class="muted">no stock plate</span><b>{s['not_manufacturable']}</b></div>
<div class="tile bad"><span class="muted">false passes caught</span><b>{s['false_passes_caught']}</b></div>
</div>
<h2>Selected design</h2>{selected}
<h2>Every proposal</h2>
<p class="muted">Values are claimed / verified. Worst case is +{req.load_uncertainty:.0%} load and a {req.thickness_tolerance_mm:g} mm thinner plate.
A red row is a design the proposer said would pass and the physics says would not.</p>
<div class="scroll"><table><thead><tr><th>id</th><th>material</th><th>w x t, mm</th><th>yield MS</th><th>worst-case MS</th>
<th>deflection, mm</th><th>mass, g</th><th>label</th><th>reason and findings</th></tr></thead><tbody>
{rows}
</tbody></table></div>"""
    return _page(f"Design Trace {req.case}", body)


def render_eval(results: list[CaseResult]) -> str:
    def pct(x):
        return "-" if x is None else f"{x * 100:.0f}%"

    def money(x):
        return "-" if x is None else f"${x:.2f}"

    board = "\n".join(
        f"<tr><td><b>{escape(b['proposer'])}</b></td><td>{b['solved']} / {b['cases']}</td><td>{b['errors']}</td><td>{b['proposals']}</td>"
        f"<td>{pct(b['verified_rate'])}</td><td>{pct(b['false_pass_rate'])}</td><td>{pct(b['mean_regret'])}</td>"
        f"<td>{b['tool_calls']}</td><td>{b['tokens']:,}</td><td>{money(b['cost_usd'])}</td><td>{b['seconds']:.0f}</td></tr>"
        for b in scoreboard(results))

    detail = []
    for r in results:
        if r.error:
            detail.append(f"<tr><td>{r.case}</td><td>{escape(r.proposer)}</td><td colspan='5' class='f-false_pass'>error, not scored: {escape(r.error)}</td></tr>")
            continue
        s = summarize(r.trace)
        sel = next((x for x in r.trace.records if x.proposal.id == r.trace.selected_id), None)
        chosen = f"{sel.proposal.design.material} {sel.proposal.design.width_mm:g} x {sel.proposal.design.thickness_mm:g}" if sel else "none"
        detail.append(f"<tr><td>{r.case}</td><td>{escape(r.proposer)}</td><td>{escape(chosen)}</td><td>{_num(r.best_g, '.0f')}</td>"
                      f"<td>{_num(r.optimum_g, '.0f')}</td><td>{pct(r.regret)}</td>"
                      f"<td>{s['verified']} / {s['proposals']} verified, {s['false_passes_caught']} false passes</td></tr>")

    body = f"""<h1>Do design agents earn their cost?</h1>
<p class="muted">Every proposer runs on every case. A case is solved when at least one proposal passes all seven requirements.
Regret is how much heavier the chosen design is than the true optimum found by brute force.
A false pass is a design the proposer claimed would pass that fails strength or stiffness. Crashes are recorded and left out of the scores.</p>
<h2>Scoreboard</h2>
<div class="scroll"><table><thead><tr><th>proposer</th><th>solved</th><th>errors</th><th>proposals</th><th>verified</th>
<th>false pass rate</th><th>mean regret</th><th>tool calls</th><th>tokens</th><th>cost</th><th>seconds</th></tr></thead>
<tbody>{board}</tbody></table></div>
<h2>By case</h2>
<div class="scroll"><table><thead><tr><th>case</th><th>proposer</th><th>chosen design</th><th>mass, g</th><th>optimum, g</th>
<th>regret</th><th>proposals</th></tr></thead><tbody>{''.join(detail)}</tbody></table></div>"""
    return _page("Design Agent Eval", body)
