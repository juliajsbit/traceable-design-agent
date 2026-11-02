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
