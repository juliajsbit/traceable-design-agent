import json
from types import SimpleNamespace

import pytest

from design_agent.cases import CASES, oracle
from design_agent.evals import run_eval, scoreboard
from design_agent.physics import check
from design_agent.pipeline import run, summarize
from design_agent.proposers import AgentProposer, LLMProposer, NaiveProposer, build_prompt
from design_agent.report import render, render_eval
from design_agent.schemas import Design, ProposalBatch, ProposalDraft, Requirement, Usage

REQ = Requirement()
GOOD = dict(material="Al 6061-T6", width=20, thickness=25)  # passes every requirement in case A


def draft(material="Al 6061-T6", width=20, thickness=25, ms=0.4, deflection=1.3, mass=200):
    return ProposalDraft(design=Design(material=material, width_mm=width, thickness_mm=thickness),
                         claimed_margin_yield=ms, claimed_deflection_mm=deflection, claimed_mass_g=mass, rationale="test")


class Scripted:
    """A proposer that returns fixed drafts, one list per round."""
    name, model = "scripted", None

    def __init__(self, *rounds):
        self.rounds = list(rounds)
        self.seen_feedback = []
        self.usage = Usage()

    def propose(self, req, feedback):
        self.seen_feedback.append(feedback)
        return self.rounds.pop(0)


def labels(trace):
    return [r.label for r in trace.records]


# --- physics and the verification matrix ---

def test_beam_formulas_match_a_hand_calculation():
    c = check(Design(material="Al 6061-T6", width_mm=30, thickness_mm=20), REQ)
    # stress = 6 * 2000 N * 0.15 m / (0.03 m * 0.02^2 m^2) = 150 MPa
    assert c.nominal.stress_mpa == pytest.approx(150)
    # MS yield = 276 / (1.25 * 150) - 1 = 0.472; MS ultimate = 310 / (1.4 * 150) - 1 = 0.476
    assert c.nominal.margin_yield == pytest.approx(0.472)
    assert c.nominal.margin_ultimate == pytest.approx(0.476, abs=1e-3)
    # deflection = 4 * 2000 * 0.15^3 / (68.9e9 * 0.03 * 0.02^3) = 1.633 mm; mass = 2700 * 0.03 * 0.02 * 0.15 = 243 g
    assert c.nominal.deflection_mm == pytest.approx(1.633, abs=1e-3)
    assert c.mass_g == pytest.approx(243)


def test_worst_case_uses_more_load_and_a_thinner_plate():
    c = check(Design(material="Al 6061-T6", width_mm=30, thickness_mm=20), REQ)
    assert c.worst_case.load_n == pytest.approx(2200) and c.worst_case.thickness_mm == pytest.approx(19.5)
    assert c.worst_case.margin_yield < c.nominal.margin_yield


def test_the_matrix_has_one_row_per_requirement():
    c = check(Design(material="Al 6061-T6", width_mm=30, thickness_mm=20), REQ)
    assert [r.id for r in c.results] == [f"REQ-00{i}" for i in range(1, 8)]
    assert c.passes


# --- labels: why a design was rejected ---

def test_a_false_pass_is_caught_and_rejected():
    # 30 x 12 mm aluminium really has a yield margin of -0.47, the proposer claims +1.0
    trace = run(REQ, Scripted([draft(width=30, thickness=12, ms=1.0, deflection=2.0, mass=146)]))
    r = trace.records[0]
    assert r.status == "rejected" and r.label == "fails"
    assert r.findings[0].kind == "false_pass"
    assert trace.selected_id is None


def test_a_design_that_passes_only_at_nominal_is_fragile():
    # 14 x 25 mm aluminium: positive yield margin nominal, but below zero at +10% load and a thinner plate
    trace = run(REQ, Scripted([draft(width=14, thickness=25, ms=0.07, deflection=1.8, mass=142)]))
    assert labels(trace) == ["fragile"]


def test_a_thickness_you_cannot_buy_is_not_manufacturable():
    trace = run(REQ, Scripted([draft(width=20, thickness=22, ms=0.1, deflection=1.9, mass=178)]))
    assert labels(trace) == ["not manufacturable"]


def test_designs_outside_the_envelope_are_rejected_without_a_physics_check():
    trace = run(REQ, Scripted([draft(width=200)]))
    assert trace.records[0].check is None and labels(trace) == ["out of envelope"]


def test_tall_thin_blades_fail_the_proportions_check():
    trace = run(REQ, Scripted([draft(width=10, thickness=30, ms=0.4, deflection=1.6, mass=122)]))
    r = trace.records[0]
    assert r.label == "fails" and "REQ-007" in r.reason


# --- selection and the loop ---

def test_selection_picks_the_lightest_verified_design_and_explains_lighter_rejections():
    heavy = draft(material="Steel 4130", width=35, thickness=25, ms=2.0, deflection=0.3, mass=1030)
    light = draft(**{"width": 20, "thickness": 25})
    lying = draft(width=30, thickness=12, ms=1.0, deflection=2.0, mass=146)
    trace = run(REQ, Scripted([heavy, light, lying]))
    assert trace.selected_id == "r1-02"
    assert "r1-03" in trace.selection_reason and "fails" in trace.selection_reason


def test_the_naive_proposer_forgets_bending_and_the_verifier_notices():
    s = summarize(run(REQ, NaiveProposer()))
    assert s["false_passes_caught"] > 0 and s["claims_off"] > 0


def test_rejections_are_fed_back_to_the_next_round():
    proposer = Scripted([draft(width=30, thickness=12, ms=1.0)], [draft()])
    trace = run(REQ, proposer, rounds=2)
    assert proposer.seen_feedback[0] == []
    assert any("rejected" in line for line in proposer.seen_feedback[1])
    assert [r.proposal.id for r in trace.records] == ["r1-01", "r2-01"]


# --- model proposers, with a fake client so no test calls the API ---

def response(stop_reason="end_turn", content=(), parsed=None, tokens=(100, 50)):
    return SimpleNamespace(stop_reason=stop_reason, content=list(content), parsed_output=parsed,
                           usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]), stop_details=None)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(parse=self._next, create=self._next))

    def _next(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_llm_proposer_returns_the_parsed_batch_and_counts_cost():
    client = FakeClient(response(parsed=ProposalBatch(proposals=[draft()]), tokens=(1000, 500)))
    proposer = LLMProposer(client=client)
    trace = run(REQ, proposer)
    assert trace.selected_id == "r1-01"
    assert proposer.usage.cost_usd == pytest.approx((1000 * 4 + 500 * 20) / 1e6)
    assert client.calls[0]["fallbacks"] == "default"


def test_agent_calls_the_physics_tool_then_returns_proposals():
    tool_call = SimpleNamespace(type="tool_use", id="t1", name="check_design",
                                input={"material": "Al 6061-T6", "width_mm": 20, "thickness_mm": 25})
    final = SimpleNamespace(type="text", text=ProposalBatch(proposals=[draft()]).model_dump_json())
    client = FakeClient(response("tool_use", [tool_call]), response("end_turn", [final]))
    proposer = AgentProposer(client=client)
    trace = run(REQ, proposer)

    assert proposer.usage.tool_calls == 1 and proposer.usage.llm_calls == 2
    tool_result = client.calls[1]["messages"][-1]["content"][0]
    assert tool_result["tool_use_id"] == "t1" and json.loads(tool_result["content"])["passes"] is True
    assert trace.selected_id == "r1-01"


def test_agent_gives_up_after_the_turn_limit():
    tool_call = SimpleNamespace(type="tool_use", id="t1", name="check_design",
                                input={"material": "Al 6061-T6", "width_mm": 20, "thickness_mm": 25})
    client = FakeClient(*[response("tool_use", [tool_call]) for _ in range(3)])
    with pytest.raises(RuntimeError, match="did not finish"):
        AgentProposer(client=client, max_turns=3).propose(REQ, [])


def test_prompt_states_every_requirement():
    prompt = build_prompt(REQ, [], 8, with_tool=True)
    assert all(f"REQ-00{i}" in prompt for i in range(1, 8)) and "check_design" in prompt


# --- eval and reports ---

def test_every_case_has_a_feasible_optimum():
    for req in CASES.values():
        best = oracle(req)
        assert best is not None and check(best[0], req).passes


def test_eval_scores_regret_against_the_optimum_and_keeps_errors_out():
    class Boom:
        name, model, usage = "boom", None, Usage()

        def propose(self, req, feedback):
            raise RuntimeError("model unavailable")

    results = run_eval({"A": CASES["A"]}, {"naive": NaiveProposer, "boom": Boom}, concurrency=2)
    board = {row["proposer"]: row for row in scoreboard(results)}
    assert board["boom"]["errors"] == 1 and board["boom"]["proposals"] == 0
    naive = next(r for r in results if r.proposer == "naive")
    assert naive.regret is not None and naive.regret >= 0
    assert "model unavailable" in render_eval(results)


def test_report_shows_the_matrix_and_every_proposal():
    trace = run(REQ, NaiveProposer())
    html = render(trace)
    assert "Verification matrix" in html and "REQ-006" in html
    assert html.count("<tr class=") == len(trace.records)


def test_rescore_scores_saved_traces_again_and_keeps_errors():
    from design_agent.evals import CaseResult, rescore
    trace = run(CASES["A"], Scripted([draft(width=15, thickness=25, ms=0.0, deflection=2.5, mass=152)]))
    stale = CaseResult(case="A", proposer="scripted", trace=trace, optimum_g=1.0, best_g=1.0, regret=-0.5)
    failed = CaseResult(case="A", proposer="boom", error="model unavailable")
    fresh, kept = rescore([stale, failed])
    assert fresh.regret == pytest.approx(0) and kept.error == "model unavailable"


# --- the LangGraph agent, with a fake chat model so no test calls the API ---

class FakeChat:
    def __init__(self, *messages, final=None):
        self.messages = list(messages)
        self.final = final
        self.tools = None

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return self.messages.pop(0)

    def with_structured_output(self, schema, include_raw=False, **kwargs):
        final = self.final
        return SimpleNamespace(invoke=lambda messages: {"raw": final, "parsed": schema(proposals=[draft()]), "parsing_error": None})


def ai(content="", tool_calls=(), tokens=(100, 50)):
    from langchain_core.messages import AIMessage
    return AIMessage(content=content, tool_calls=list(tool_calls),
                     usage_metadata={"input_tokens": tokens[0], "output_tokens": tokens[1], "total_tokens": sum(tokens)},
                     response_metadata={"model_name": "claude-haiku-5-5", "stop_reason": "end_turn"})


def test_langgraph_agent_calls_the_physics_tool_then_finalizes():
    from design_agent.graph_agent import LangGraphProposer
    call = {"name": "check_design", "id": "t1", "type": "tool_call",
            "args": {"material": "Al 6061-T6", "width_mm": 20, "thickness_mm": 25}}
    chat = FakeChat(ai(tool_calls=[call]), ai("done"), final=ai("{}", tokens=(10, 5)))
    proposer = LangGraphProposer(model="claude-haiku-5-5", chat_model=chat)
    trace = run(REQ, proposer)
    assert trace.selected_id == "r1-01"
    assert proposer.usage.tool_calls == 1 and proposer.usage.llm_calls == 3
    assert proposer.usage.input_tokens == 210 and proposer.usage.served_by == ["claude-haiku-5-5"]


def test_langgraph_agent_stops_at_the_turn_limit():
    from langgraph.errors import GraphRecursionError
    from design_agent.graph_agent import LangGraphProposer
    call = {"name": "check_design", "id": "t1", "type": "tool_call",
            "args": {"material": "Al 6061-T6", "width_mm": 20, "thickness_mm": 25}}
    chat = FakeChat(*[ai(tool_calls=[call]) for _ in range(10)], final=ai("{}"))
    with pytest.raises(GraphRecursionError):
        LangGraphProposer(model="claude-haiku-5-5", chat_model=chat, max_turns=3).propose(REQ, [])


def test_team_plans_runs_three_specialists_in_parallel_and_verifies():
    from design_agent.team import Plan, run_team

    class TeamChat(FakeChat):
        def with_structured_output(self, schema, include_raw=False, **kwargs):
            if schema is Plan:
                plan = Plan(reasoning="stiffness governs", assignments=[])
                return SimpleNamespace(invoke=lambda prompt: {"raw": ai("{}"), "parsed": plan, "parsing_error": None})
            return super().with_structured_output(schema, include_raw)

    call = {"name": "check_design", "id": "t1", "type": "tool_call",
            "args": {"material": "Al 6061-T6", "width_mm": 20, "thickness_mm": 25}}
    events = []
    run_team(REQ, "claude-haiku-5-5", events.append, n=1,
             make_chat=lambda: TeamChat(ai(tool_calls=[call]), ai("done"), final=ai("{}")),
             market=False, suppliers=False, review=False, red=False, chief=False)

    kinds = [e["type"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    starts = {e["agent"] for e in events if e["type"] == "agent_start"}
    assert starts == {"lead", "Al 6061-T6", "Ti-6Al-4V", "Steel 4130", "compliance", "fragility", "chief"}
    assert next(e for e in events if e["type"] == "review")["decision"] == "ship"
    assert next(e for e in events if e["type"] == "fragility")["samples"] == 2000
    assert next(e for e in events if e["type"] == "compliance")["summary"]["pass"] > 0
    assert kinds.count("check") == 3 and kinds.count("verified") == 3
    decision = next(e for e in events if e["type"] == "decision")
    assert decision["selected"]["mass_g"] == pytest.approx(202.5) and decision["optimum"]["mass_g"] > 0
    assert all("t" in e for e in events)


def test_replay_streams_a_saved_recording(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import design_agent.server as server
    monkeypatch.setattr(server, "RECORDINGS", tmp_path)
    (tmp_path / "demo.jsonl").write_text('{"type": "run_start", "t": 0}\n{"type": "run_end", "t": 0.01, "seconds": 0.01}\n')
    client = TestClient(server.app)
    assert client.get("/api/info").json()["recordings"] == ["demo"]
    body = client.get("/api/replay", params={"name": "demo", "speed": 4}).text
    assert body.count("data: ") == 2 and "run_end" in body
    assert client.get("/api/replay", params={"name": "../secrets"}).status_code == 404


def test_compliance_engine_checks_rules_in_code():
    from design_agent.compliance import evaluate, summary, supplier_requirements
    steel = Design(material="Steel 4130", width_mm=20, thickness_mm=25)
    items = {i["id"]: i for i in evaluate(steel, REQ, check(steel, REQ))}
    assert items["S-01"]["status"] == "pass" and items["S-01"]["margin"] > 0
    assert items["P-01"]["status"] == "action" and items["M-04"]["status"] == "action"   # specialty metal, needs a finish
    assert "AMS 6350" in items["M-06"]["value"] and items["M-01"]["status"] == "action"
    assert supplier_requirements(steel) == {"as9100": True, "dfars_melt": True}
    weak = Design(material="Al 6061-T6", width_mm=30, thickness_mm=12)
    weak_items = evaluate(weak, REQ, check(weak, REQ))
    assert {i["id"]: i for i in weak_items}["S-01"]["status"] == "fail" and summary(weak_items)["fail"] >= 1
    assert supplier_requirements(weak)["dfars_melt"] is False


def test_supplier_desk_checks_eligibility_estimates_and_ranks():
    from design_agent import suppliers as sd
    vendors = {v["id"]: v for v in sd.VENDORS}
    ti = Design(material="Ti-6Al-4V", width_mm=12, thickness_mm=20)
    needs = {"as9100": True, "dfars_melt": True}
    assert sd.eligibility(vendors["A"], ti, needs) == []
    assert "does not work with Ti-6Al-4V" in sd.eligibility(vendors["B"], ti, needs)
    assert "no AS9100 quality system" in sd.eligibility(vendors["C"], ti, needs)
    assert "cannot certify a DFARS compliant melt source" in sd.eligibility(vendors["D"], ti, needs)
    est = sd.estimate(vendors["A"], ti, CASES["E"], 10)
    # blank 22 x 150 x 20 mm of titanium: 66 cm3 -> 0.29 kg -> about $19 of material
    assert est["material_usd"] == pytest.approx(22 * 150 * 20 / 1e9 * 4430 * 65, rel=1e-3)
    assert 50 < est["price_each_usd"] < 500 and est["plate_in_stock"] is True
    quotes = [{"vendor": v, "name": v, "quote": {"bid": True, "price_each_usd": p, "lead_time_days": d}, "eligible": True,
               "flags": [], "estimate": {"lead_time_days": d}} for v, p, d in [("A", 300, 20), ("X", 200, 30), ("Y", 250, 10)]]
    quotes.append({"vendor": "Z", "name": "Z", "quote": {"bid": True, "price_each_usd": 100, "lead_time_days": 5},
                   "eligible": False, "flags": [], "estimate": {"lead_time_days": 5}})
    ranked = sd.rank(quotes, 10)
    assert "Z" not in [r["vendor"] for r in ranked] and ranked[0]["vendor"] == "Y"   # cheapest-and-fastest mix wins


def test_vendor_agent_quote_is_checked_against_the_shop_estimate():
    from design_agent import suppliers as sd
    vendor = {v["id"]: v for v in sd.VENDORS}["C"]                 # not AS9100
    alu = Design(material="Al 6061-T6", width_mm=20, thickness_mm=25)

    class QuoteChat(FakeChat):
        def with_structured_output(self, schema, include_raw=False, **kwargs):
            q = sd.Quote(bid=True, price_each_usd=1.0, lead_time_days=3, notes="cheap")
            return SimpleNamespace(invoke=lambda m: {"raw": ai("{}"), "parsed": q, "parsing_error": None})

    call = {"name": "estimate_job", "id": "e1", "type": "tool_call", "args": {"quantity": 10}}
    steps = []
    result = sd.ask_vendor(vendor, alu, REQ, 10, {"as9100": True, "dfars_melt": False},
                           QuoteChat(ai(tool_calls=[call]), ai("quoting")), steps.append)
    assert [s["tools"] for s in steps] == [["estimate_job"], []]
    assert any("bid rejected" in f for f in result["flags"]) and any("does not match" in f for f in result["flags"])
    assert result["eligible"] is False


def test_vendors_without_a_search_result_link_are_dropped():
    from design_agent.sourcing import find_vendors
    result = SimpleNamespace(url="https://www.realshop.com/cnc", title="Real Shop")
    listed = [{"name": "Real Shop", "location": "Gardena, CA", "url": "https://realshop.com/about", "capabilities": "CNC", "certifications": "AS9100D"},
              {"name": "Made Up Inc", "location": "Nowhere", "url": "https://madeup.example", "capabilities": "CNC", "certifications": "AS9100"}]
    content = [SimpleNamespace(type="server_tool_use", name="web_search", input={"query": "AS9100 CNC"}),
               SimpleNamespace(type="web_search_tool_result", content=[result]),
               SimpleNamespace(type="text", text=f"<vendors>{json.dumps(listed)}</vendors>")]
    usage = SimpleNamespace(input_tokens=1000, output_tokens=100, server_tool_use=SimpleNamespace(web_search_requests=1))
    client = FakeClient(SimpleNamespace(stop_reason="end_turn", content=content, usage=usage))
    found = find_vendors(Design(material="Steel 4130", width_mm=69, thickness_mm=8), CASES["F"], "claude-haiku-5-5", client=client)
    assert [v["name"] for v in found["vendors"]] == ["Real Shop"] and found["dropped"] == 1
    assert found["queries"] == ["AS9100 CNC"] and found["usage"]["cost_usd"] == pytest.approx(0.01 + (1000 * 0.10 + 100 * 0.50) / 1e6)
    assert client.calls[0]["tools"][0]["type"] == "web_search_20250305"


def test_requirements_desk_does_the_arithmetic_and_flags_assumptions():
    from design_agent.requirements_desk import RequirementDraft, build
    draft = RequirementDraft(part="star tracker bracket", payload_mass_kg=4, design_load_g=15, length_mm=150,
                             max_deflection_mm=1.5, thickness_max_mm=8, sources=[], open_questions=[])
    req, derived, assumed, problems = build(draft)
    assert req.load_n == pytest.approx(4 * 15 * 9.81, abs=0.1) and "589 N" in derived[0]
    assert req.thickness_range_mm == (3, 8) and any("width_max_mm" in a for a in assumed) and problems == []
    bad = RequirementDraft(part="x", width_max_mm=5, sources=[], open_questions=[])
    assert any("width limit" in p for p in build(bad)[3])


def test_red_team_attacks_are_rechecked_by_code():
    from design_agent.redteam import RedTeamReport, Attack, breaking_point, red_team, survives
    good = Design(material="Al 6061-T6", width_mm=20, thickness_mm=25)
    bp = breaking_point(good, REQ)
    assert bp["load_multiplier"] > 1 and survives(good, REQ, load_multiplier=bp["load_multiplier"] * 1.05)["survives"] is False
    attacks = [Attack(title="harder launch", rationale="r", testable=True, load_multiplier=3, expected_to_fail=False, severity="high"),
               Attack(title="fatigue", rationale="r", testable=False, expected_to_fail=True, severity="medium")]

    class RedChat(FakeChat):
        def with_structured_output(self, schema, include_raw=False, **kwargs):
            return SimpleNamespace(invoke=lambda m: {"raw": ai("{}"), "parsed": RedTeamReport(attacks=attacks), "parsing_error": None})

    report = red_team(good, REQ, RedChat(ai("no tests needed")))
    first, second = report["attacks"]
    assert first["result"] == "confirmed" and first["agent_right"] is False    # it said it would survive x3; code says no
    assert second["result"] == "needs analysis" and report["breaking_point"] == bp


def test_brief_run_stops_for_approval_and_uses_the_edits():
    from design_agent.requirements_desk import RequirementDraft
    from design_agent.team import Plan, run_team
    req_draft = RequirementDraft(part="bracket", payload_mass_kg=4, design_load_g=15, length_mm=150, max_deflection_mm=3,
                                 sources=[], open_questions=["which launcher?"])

    class BriefChat(FakeChat):
        def with_structured_output(self, schema, include_raw=False, **kwargs):
            parsed = {RequirementDraft: req_draft, Plan: Plan(reasoning="ok", assignments=[])}.get(schema) or schema(proposals=[draft()])
            return SimpleNamespace(invoke=lambda m: {"raw": ai("{}"), "parsed": parsed, "parsing_error": None})

    asked, events = [], []
    run_team(None, "claude-haiku-5-5", events.append, n=1, make_chat=lambda: BriefChat(ai("done"), final=ai("{}")),
             market=False, suppliers=False, review=False, red=False, chief=False, brief="a 4 kg tracker",
             approve=lambda payload: asked.append(payload) or {"edits": {"load_n": 1000}})
    kinds = [e["type"] for e in events]
    assert kinds.index("requirements") < kinds.index("approval_needed") < kinds.index("approved") < kinds.index("design_start")
    assert asked[0]["requirement"]["load_n"] == pytest.approx(588.6, abs=0.1)
    assert next(e for e in events if e["type"] == "design_start")["requirement"]["load_n"] == 1000


def test_monte_carlo_finds_the_tight_design_fragile_and_is_repeatable():
    from design_agent.fragility import monte_carlo, tradeoffs
    tight = Design(material="Al 6061-T6", width_mm=15, thickness_mm=25)    # worst-case MS about +0.00 in case A
    roomy = Design(material="Al 6061-T6", width_mm=20, thickness_mm=25)
    a, b = monte_carlo(tight, REQ), monte_carlo(roomy, REQ)
    assert a["p_failure"] > b["p_failure"] and a == monte_carlo(tight, REQ)
    assert abs(a["sensitivity"]["load"]) > abs(a["sensitivity"]["width_mm"])      # load scatter matters more than width
    assert sum(a["histogram"]["counts"]) == a["samples"]
    rows = tradeoffs([{"id": "t", "design": tight.model_dump()}, {"id": "r", "design": roomy.model_dump()}], REQ)
    assert rows[0]["mass_g"] < rows[1]["mass_g"] and rows[0]["cost_usd"] > 0


# --- Chief Engineer review and the redesign loop ---

TIGHT = Design(material="Al 6061-T6", width_mm=15, thickness_mm=25)   # verified, but fails in about 2% of sampled parts
ROOMY = Design(material="Al 6061-T6", width_mm=20, thickness_mm=25)


def test_review_gate_blocks_a_fragile_design_and_code_has_the_last_word():
    from design_agent.review import ReviewDecision, decide, gate, select_robust
    tight, roomy = gate(TIGHT, REQ), gate(ROOMY, REQ)
    assert [f["id"] for f in tight if f["blocking"]] == ["G-01"] and not any(f["blocking"] for f in roomy)

    ship = ReviewDecision(decision="ship", rationale="looks fine")
    v = decide(TIGHT, tight, ship, round_=1)
    assert v["decision"] == "redesign" and v["overruled"] and "G-01" in v["directive"]      # cannot ship past a blocker
    again = ReviewDecision(decision="redesign", rationale="more margin", directive="aim for +0.10 worst-case margin")
    assert decide(TIGHT, tight, again, round_=1)["directive"] == "aim for +0.10 worst-case margin"
    assert decide(TIGHT, tight, again, round_=2)["decision"] == "escalate"                   # no rounds left
    v = decide(ROOMY, roomy, again, round_=1)
    assert v["decision"] == "ship" and v["rule"] == "a redesign needs a blocking finding"
    assert decide(TIGHT, tight, None, round_=1)["decision"] == "redesign"                    # works without the agent

    pool = [{"id": "t", "design": TIGHT.model_dump(), "mass_g": 152}, {"id": "r", "design": ROOMY.model_dump(), "mass_g": 202}]
    assert select_robust(pool, REQ)["id"] == "r" and select_robust(pool[:1], REQ) is None


def loop_chat(rounds: list[Design], decision="ship"):
    """A chat factory for the team: specialists propose rounds[i] in design round i + 1, the chief always says `decision`."""
    from design_agent.review import ReviewDecision
    from design_agent.team import Plan
    seen = {"plans": 0}

    class LoopChat(FakeChat):
        def with_structured_output(self, schema, include_raw=False, **kwargs):
            if schema is Plan:
                seen["plans"] += 1
                parsed = Plan(reasoning="ok", assignments=[])
            elif schema is ReviewDecision:
                parsed = ReviewDecision(decision=decision, rationale="r")
            elif schema.__name__ in ("Quote", "Memo"):
                parsed = schema(bid=True, price_each_usd=1.0) if schema.__name__ == "Quote" else schema(memo="buy from A")
            else:
                d = rounds[min(seen["plans"], len(rounds)) - 1]
                parsed = schema(proposals=[draft(width=d.width_mm, thickness=d.thickness_mm)])
            return SimpleNamespace(invoke=lambda m: {"raw": ai("{}"), "parsed": parsed, "parsing_error": None})

    return lambda: LoopChat(ai("done"), final=ai("{}"))


def test_a_blocked_design_goes_back_to_the_team_and_the_second_round_ships():
    from design_agent.team import run_team
    events = []
    run_team(REQ, "claude-haiku-5-5", events.append, n=1, make_chat=loop_chat([TIGHT, ROOMY]),
             market=False, suppliers=False, review=False, red=False)
    kinds = [e["type"] for e in events]
    first, second = [e for e in events if e["type"] == "review"]
    assert first["decision"] == "redesign" and first["proposed"] == "ship" and first["overruled"]
    assert second["decision"] == "ship" and second["round"] == 2 and not second["overruled"]
    assert kinds.count("redesign_start") == 1 and kinds.count("plan") == 2 and kinds.count("compliance") == 1
    assert kinds.index("redesign_start") > kinds.index("review")
    picks = [e["selected"] for e in events if e["type"] == "decision"]
    assert picks[0]["design"]["width_mm"] == 15 and picks[1]["design"]["width_mm"] == 20 and picks[1]["id"] == "Al-r2-1"
    assert {e["id"] for e in events if e["type"] == "verified"} >= {"Al-1", "Al-r2-1"}


def test_review_escalates_when_rounds_run_out_and_holds_the_purchase():
    from design_agent.team import run_team
    events = []
    run_team(REQ, "claude-haiku-5-5", events.append, n=1, make_chat=loop_chat([TIGHT], decision="redesign"),
             market=False, suppliers=True, review=False, red=False)
    kinds = [e["type"] for e in events]
    reviews = [e for e in events if e["type"] == "review"]
    assert [r["decision"] for r in reviews] == ["redesign", "escalate"] and reviews[1]["rule"] == "no rounds left"
    assert "no verified design meets the review target" in [e for e in events if e["type"] == "decision"][1]["reason"]
    assert "compliance" in kinds and "rfq" not in kinds and "quote" not in kinds


# --- change impact: a run that starts from an earlier run with a changed requirement ---

def baseline_run():
    import json
    from design_agent.team import run_team
    snap = run_team(REQ, "claude-haiku-5-5", lambda e: None, n=1, make_chat=loop_chat([ROOMY]), market=False, review=False, red=False)
    return json.loads(json.dumps(snap))   # a snapshot is plain JSON, as the server saves it


def change_run(base, make_chat=None, **edits):
    from design_agent.impact import apply_edits
    from design_agent.team import run_team
    events, qty = [], edits.pop("qty", 10)
    old = Requirement(**base["req"])   # as the server does: the requirement comes back from the saved JSON
    run_team(apply_edits(old, edits), "claude-haiku-5-5", events.append, n=1, make_chat=make_chat or loop_chat([ROOMY]),
             market=False, review=False, red=False, baseline=base, qty=qty)
    steps = {e["step"]: e for e in events if e["type"] == "step"}
    return events, steps, next(e for e in events if e["type"] == "impact")


def no_model():
    raise AssertionError("a model was called")


def test_diff_says_which_fields_changed_and_which_way():
    from design_agent.impact import apply_edits, diff
    changes = diff(REQ, apply_edits(REQ, {"load_n": 2400, "max_deflection_mm": 4, "width_max_mm": 60}))
    assert [(c["field"], c["effect"]) for c in changes] == [("load_n", "tighter"), ("max_deflection_mm", "looser")]
    with pytest.raises(ValueError):
        apply_edits(REQ, {"width_max_mm": 5})


def test_a_change_run_with_no_change_reuses_every_step_and_calls_no_model():
    base = baseline_run()
    events, steps, impact = change_run(base, make_chat=no_model)
    assert impact["search"] == "kept" and impact["changes"] == []
    assert {s["action"] for s in steps.values()} == {"reused"} and len(steps) == 10     # 6 steps + 4 vendors
    kinds = [e["type"] for e in events]
    assert "plan" not in kinds and "usage" not in kinds and kinds.count("quote") == 4
    assert next(e for e in events if e["type"] == "decision")["selected"]["design"] == ROOMY.model_dump()
    assert all(e["reused"] for e in events if e["type"] in ("quote", "review", "compliance"))


def test_a_tighter_load_keeps_the_search_and_reruns_only_the_steps_that_read_the_load():
    base = baseline_run()
    events, steps, impact = change_run(base, load_n=2100)
    assert impact["search"] == "kept" and impact["still_verified"] == impact["were_verified"] == 3
    assert "plan" not in [e["type"] for e in events]                                  # no lead, no specialists
    rerun = {k for k, s in steps.items() if s["action"] == "rerun"}
    assert rerun == {"redteam", "fragility", "chief", "compliance"} and steps["chief"]["changed"] == ["req.load_n"]
    assert all(steps[f"vendor-{v}"]["action"] == "reused" for v in "ABCD") and steps["procurement"]["action"] == "reused"


def test_a_change_that_breaks_the_selected_design_sends_the_team_back_with_what_still_works():
    from design_agent.team import run_team
    import json
    both = json.loads(json.dumps(run_team(REQ, "claude-haiku-5-5", lambda e: None, n=1, make_chat=loop_chat([TIGHT]),
                                           market=False, suppliers=False, review=False, red=False, max_rounds=1)))
    events, steps, impact = change_run(both, make_chat=loop_chat([ROOMY]), load_n=2300)
    assert impact["search"] == "rerun" and "no longer passes" in impact["why"] and impact["still_verified"] == 0
    assert impact["pool"][0]["before"] == "verified" and impact["pool"][0]["after"] != "verified"
    ids = [e["id"] for e in events if e["type"] == "verified"]
    assert ids[:3] == ["Al-1", "Ti-1", "Steel-1"] and "Al-r2-1" in ids              # old proposals rechecked, then the new round
    picked = [e for e in events if e["type"] == "decision"][-1]
    assert picked["selected"]["id"] == "Al-r2-1" and picked["round"] == 1            # a change starts a new review cycle
    assert not any(s["action"] == "reused" for s in steps.values())


def test_a_looser_requirement_searches_again_and_a_new_quantity_reruns_only_the_quotes():
    base = baseline_run()
    events, steps, impact = change_run(base, max_deflection_mm=4)
    assert impact["search"] == "rerun" and "looser" in impact["why"] and "plan" in [e["type"] for e in events]

    events, steps, impact = change_run(base, qty=50)
    assert impact["search"] == "kept" and impact["changes"][0]["field"] == "qty"
    assert {k for k, s in steps.items() if s["action"] == "rerun"} == {"fragility", "vendor-A", "vendor-B", "vendor-C", "vendor-D", "procurement"}
    assert steps["vendor-A"]["changed"] == ["qty"] and steps["chief"]["action"] == "reused"


def test_server_starts_a_change_run_only_from_a_saved_snapshot_with_sane_edits(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import design_agent.server as server
    from design_agent.impact import load, save
    monkeypatch.setattr(server, "RECORDINGS", tmp_path)
    save(tmp_path, "base", {"req": REQ.model_dump(), "qty": 10, "results": [], "steps": {}})
    client = TestClient(server.app)
    assert client.get("/api/info").json()["snapshots"] == ["base"]
    assert client.get("/api/run", params={"base": "missing"}).status_code == 404
    assert client.get("/api/run", params={"base": "base", "edits": '{"width_max_mm": 1}'}).status_code == 400
    assert client.get("/api/run", params={"base": "base", "edits": '{"qty": 0}'}).status_code == 400
    assert load(tmp_path, "../base") is None and load(tmp_path, "base")["run_id"] == "base"


def test_a_value_hashes_the_same_before_and_after_it_is_saved_as_json():
    import json
    from design_agent.impact import fingerprint
    inputs = {"req.width_range_mm": (10, 60), "qty": 10, "on": True, "design": {"width_mm": 12}}
    assert fingerprint(inputs) == fingerprint(json.loads(json.dumps(inputs))) == fingerprint({**inputs, "qty": 10.0})
    assert fingerprint(inputs) != fingerprint({**inputs, "qty": 11})


# --- evidence ledger and the Design Package ---

def recorded_run():
    from design_agent.team import run_team
    events = []
    run_team(REQ, "claude-haiku-5-5", events.append, n=1, make_chat=loop_chat([TIGHT, ROOMY]), market=False, review=False, red=False)
    return events


def test_the_ledger_chains_every_event_and_notices_a_changed_recording():
    import json
    from design_agent.ledger import entries, verify
    events = json.loads(json.dumps(recorded_run()))            # as read back from the recording
    chain = verify(events)
    assert chain["ok"] and chain["chained"] == len(events) and chain["head"] == events[-1]["h"]
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))

    edited = json.loads(json.dumps(events))
    target = next(e for e in edited if e["type"] == "review")
    target["decision"] = "ship"                                 # someone turns the redesign into a release
    assert verify(edited)["ok"] is False and verify(edited)["broken_at"] == target["seq"]
    assert verify(events[:5] + events[6:])["broken_at"] == 6    # a dropped event
    assert verify([{"type": "run_start"}]) == {"events": 1, "chained": 0, "ok": False, "broken_at": None, "head": None}

    rows = entries(events)
    assert {"code", "agent", "agent + code"} <= {r["kind"] for r in rows}
    assert any(r["who"] == "Chief Engineer" and "overruled" in r["what"] for r in rows)


def test_the_design_package_is_built_from_the_recording_alone():
    from design_agent.package import build
    events = recorded_run()
    page = build(events, "demo-run")
    for text in ("RELEASED", "Al 6061-T6, 20 x 25 mm", "Verification matrix", "REQ-006", "Alternatives considered", "Chief Engineer review",
                 "REDESIGN", "Code overruled it", "Compliance", "Open items", "Evidence ledger", "verified: every event matches its hash",
                 events[-1]["h"]):
        assert text in page, text
    events[3]["t"] = 99
    assert "BROKEN at event 4" in build(events)
    assert "NO DESIGN" in build([{"type": "run_start", "case": "A", "model": "m"}, {"type": "decision", "selected": None, "selected_id": None, "reason": "no design passed verification"}])


def test_the_design_package_escapes_what_agents_and_the_web_wrote(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import design_agent.server as server
    from design_agent.package import build
    market = {"type": "market", "dropped": 0, "vendors": [
        {"name": "<script>alert(1)</script>", "url": "javascript:alert(1)", "location": "LA", "capabilities": "<b>x</b>", "certifications": ""}]}
    page = build([{"type": "run_start", "case": "A", "model": "m"}, market])
    assert "<script>alert" not in page and "javascript:alert" not in page and "&lt;b&gt;x" in page
    monkeypatch.setattr(server, "RECORDINGS", tmp_path)
    (tmp_path / "demo.jsonl").write_text('{"type": "run_start", "case": "A", "model": "m", "t": 0}\n')
    client = TestClient(server.app)
    assert "Design Package" in client.get("/api/package", params={"name": "demo"}).text
    assert client.get("/api/package", params={"name": "../demo"}).status_code == 404
    assert "attachment" in client.get("/api/package", params={"name": "demo", "download": True}).headers["content-disposition"]


def test_team_input_and_retry_rules():
    import anthropic
    import httpx
    from design_agent.team import build_team, transient
    graph, _ = build_team("claude-haiku-5-5", n=1, emit=lambda e: None, make_chat=lambda: None)
    assert set(graph.get_input_jsonschema()["properties"]) == {"case", "brief", "req"}
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    assert transient(anthropic.APIConnectionError(request=request))
    assert not transient(ValueError("bad schema"))
