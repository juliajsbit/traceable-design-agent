"""A design team, from a mission brief to a part someone can make.

    START -> requirements_desk -> human_approval -> change_impact -> lead_engineer -> ...

                         +------------------------ redesign, with a directive ------------------------+
                         v                                                                            |
    ... -> lead_engineer -> material_specialist (Al) ----+                                              |
                        -> material_specialist (Ti) ----+-> verifier -> red_team_review -> monte_carlo -> chief_engineer
                        -> material_specialist (Steel) -+
    chief_engineer -> compliance_engine -> supplier_agent x4 (in parallel) -> procurement -> market_scan -> END

requirements_desk:   an agent turns a brief into a Requirement, every value with its source; code does the arithmetic
human_approval:      the graph stops with interrupt() until a person approves or edits the requirement
change_impact:       only in a change run (see below); otherwise passes through
lead_engineer:       a model reads the requirement and writes one note per material
material_specialist: the LangGraph tool agent from graph_agent.py, one per material, all running at once
verifier:            no model; recomputes every proposal and picks the lightest verified design
red_team_review:     an agent tries to break the selected design; code re-runs every attack it claims
monte_carlo:         no model; Monte Carlo on the selected design, and mass, cost, margin and risk for every verified design
chief_engineer:      code marks the blocking findings, an agent proposes ship, redesign or escalate,
                     code has the last word; a redesign goes back to the lead, an escalation holds the purchase
compliance_engine:   checks rules in code; a review agent explains the gaps and proposes actions
supplier_agent:      one agent per approved vendor answers the request for quote with its own shop tools
procurement:         code checks and ranks the quotes; a model writes the memo
market_scan:         a model with web search finds real shops outside the approved list

A run can also start from an earlier run, the baseline, with a changed requirement (impact.py). The
change_impact node then checks the old designs again and keeps the search when it can, and every later step
runs only if one of its inputs changed.

Node names are what Studio and LangSmith show. Event and agent names (emit) stay short, because the
dashboard and the saved recordings use them.

Every step is sent to emit() as a small dict. The live dashboard (server.py) draws those events.
"""
import contextvars
import operator
import time
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, Send, interrupt
from pydantic import BaseModel

from . import compliance as ce
from . import fragility as fr
from . import impact as im
from . import ledger
from . import redteam as rt
from . import requirements_desk as rd
from . import review as rv
from . import suppliers as sd
from .cases import CASES, oracle
from .graph_agent import LangGraphProposer
from .physics import MATERIALS, check
from .proposers import PRICE_PER_MTOK, requirement_text
from .schemas import Design, MaterialName, Proposal, Requirement, Usage
from .sourcing import find_vendors
from .verify import select, verify

MAX_CONCURRENCY = 4   # at most this many nodes at once (specialists, vendors); keeps the team under the API rate limit


def transient(error: Exception) -> bool:
    """Errors worth another try: rate limit, overload, timeout, a dropped connection. A bad request is not."""
    import anthropic
    return isinstance(error, (anthropic.RateLimitError, anthropic.InternalServerError,
                              anthropic.APIConnectionError, anthropic.APITimeoutError))


# Nodes whose failure stops the run get a retry with backoff. The parallel agents catch their own errors
# instead, so one failed specialist or vendor leaves a gap rather than stopping the team.
RETRY = RetryPolicy(max_attempts=3, initial_interval=2.0, retry_on=transient)

SHORT = {"Al 6061-T6": "Al", "Ti-6Al-4V": "Ti", "Steel 4130": "Steel"}


class Assignment(BaseModel):
    material: MaterialName
    note: str  # where this specialist should search, and why


class Plan(BaseModel):
    reasoning: str
    assignments: list[Assignment]


class Memo(BaseModel):
    memo: str


class TeamInput(TypedDict, total=False):
    """What a run starts from: a preset case, a mission brief, or a ready requirement. Studio shows only these."""
    case: str
    brief: str
    req: Requirement


class TeamState(TypedDict, total=False):
    case: str                 # a preset case name, as LangGraph Studio passes it
    brief: str                # or a mission brief in plain words, for the Requirements Desk
    req: Requirement
    notes: dict[str, str]
    results: Annotated[list[dict], operator.add]  # each specialist appends its own result, in parallel
    selected: dict | None
    verified: list[dict]
    attacks: list[dict]       # the Red Team's attacks on the selected design, as rechecked by code
    round: int                # design round, 1 on the first pass; the Chief Engineer can send the team back
    round_base: int           # rounds done before this review cycle began (more than 0 only in a change run)
    target: bool              # the review target applies: lightest is no longer enough
    reverify: bool            # check every proposal again, not only the new round (set by a change run)
    feedback: list[str]       # what the specialists are told when they search again after a change
    directive: str            # what the Chief Engineer asked the team to change
    rejected: Annotated[list[dict], operator.add]  # designs the review sent back
    verdict: str              # ship, redesign or escalate
    needs: dict               # what the Compliance Engine asks of suppliers
    quotes: Annotated[list[dict], operator.add]   # each vendor agent appends its quote, in parallel


def _optimum(req: Requirement, materials=None) -> dict | None:
    best = oracle(req, materials)
    return {**best[0].model_dump(), "mass_g": best[1]} if best else None


def build_team(model: str, n: int, emit, make_chat=None, max_turns: int = 12, market: bool = True, vendor_client=None,
               suppliers: bool = True, review: bool = True, qty: int = 10, red: bool = True, approval: bool = False,
               monte_carlo: bool = True, chief: bool = True, max_rounds: int = rv.MAX_ROUNDS,
               checkpointer=None, baseline: dict | None = None, steps: im.Steps | None = None):
    make_chat = make_chat or (lambda: ChatAnthropic(model=model, max_tokens=16000))
    usage: dict[str, Usage] = {}
    steps = steps or im.Steps(baseline)
    capture: contextvars.ContextVar = contextvars.ContextVar("capture", default=None)
    send = emit

    def emit(event: dict) -> None:   # pylint: disable=function-redefined  # wraps the emit argument; every node sends events through here
        buffer = capture.get()
        if buffer is not None:
            buffer.append(event)
        send(event)

    def memo(step, inputs):
        """Wrap a node so it runs only when its inputs changed since the baseline (impact.py).

        step(arg) names the step and inputs(arg) lists everything the node reads. A step with the same
        input hash as the baseline sends its saved events again, marked reused, and returns its saved state.
        """
        def wrap(fn):
            def node(arg):
                name, given = step(arg) if callable(step) else step, inputs(arg)
                saved, changed = steps.lookup(name, given)
                if saved is not None:
                    emit({"type": "step", "step": name, "action": "reused", "saved_usd": saved["cost_usd"]})
                    for e in saved["events"]:
                        if e["type"] != "usage":   # nothing was spent this time
                            emit({**e, "reused": True})
                    return saved["update"]
                if baseline:
                    emit({"type": "step", "step": name, "action": "rerun", "changed": changed})
                buffer = []
                token = capture.set(buffer)
                try:
                    update = fn(arg)
                finally:
                    capture.reset(token)
                steps.store(name, given, buffer, update)
                return update
            return node
        return wrap

    def reads(state, *fields) -> dict:
        """The selected design plus the requirement fields a step reads (all of them when none are named)."""
        req = state["req"].model_dump(exclude={"case"})
        return {"design": state.get("selected"), **{f"req.{k}": v for k, v in req.items() if not fields or k in fields}}

    def pool_of(state) -> list:
        return [v["design"] for v in state.get("verified", [])[:8]]

    def track(agent: str):
        """Count tokens and cost for one agent and send the running total to the dashboard."""
        def count(message) -> None:
            u = usage.setdefault(agent, Usage(cost_usd=0.0 if model in PRICE_PER_MTOK else None))
            meta = (message.usage_metadata or {}) if message is not None else {}
            u.llm_calls += 1
            u.input_tokens += meta.get("input_tokens", 0)
            u.output_tokens += meta.get("output_tokens", 0)
            if model in PRICE_PER_MTOK:
                price_in, price_out = PRICE_PER_MTOK[model]
                u.cost_usd += (meta.get("input_tokens", 0) * price_in + meta.get("output_tokens", 0) * price_out) / 1e6
            emit({"type": "usage", "agent": agent, **u.model_dump(exclude={"served_by"})})
        return count

    def ask(agent: str, schema, prompt):
        # json_schema: the API itself guarantees the reply matches the schema (tool calls sometimes came back malformed)
        result = make_chat().with_structured_output(schema, include_raw=True, method="json_schema").invoke(prompt)
        track(agent)(result["raw"])
        return result["parsed"]

    def requirements(state: TeamState) -> dict:
        if not state.get("brief"):
            req = state.get("req") or CASES[state.get("case", "E")]
            emit({"type": "requirements", "mode": "change" if baseline else "case", "requirement": req.model_dump()})
            return {"req": req}
        emit({"type": "agent_start", "agent": "requirements"})
        draft = ask("requirements", rd.RequirementDraft, rd.prompt(state["brief"]))
        if draft is None:
            raise RuntimeError("the requirements agent returned no draft")
        req, derived, assumed, problems = rd.build(draft)
        emit({"type": "requirements", "mode": "brief", "brief": state["brief"], "draft": draft.model_dump(),
              "requirement": req.model_dump(), "derived": derived, "assumed": assumed, "problems": problems})
        emit({"type": "agent_done", "agent": "requirements"})
        return {"req": req}

    def approve(state: TeamState) -> dict:
        if not approval or not state.get("brief"):
            return {}
        # interrupt() saves the state and stops the graph here; it resumes with what the person sends back.
        # On resume the node runs again from the top, so nothing is emitted before this line.
        decision = interrupt({"requirement": state["req"].model_dump()}) or {}
        edits = decision.get("edits") or {}
        req = im.apply_edits(state["req"], edits)
        emit({"type": "approved", "requirement": req.model_dump(), "edits": edits})
        return {"req": req}

    def impact(state: TeamState) -> dict:
        if not baseline:
            return {}
        old, new = Requirement(**baseline["req"]), state["req"]
        changes, results = im.diff(old, new), im.restore_results(baseline)
        rows = im.reverify(results, old, new)
        keep, why = im.keep_search(changes, baseline.get("selected"), new)
        if baseline.get("qty", qty) != qty:
            changes.append({"field": "qty", "old": baseline["qty"], "new": qty, "unit": "parts", "effect": "order"})
        emit({"type": "impact", "changes": changes, "pool": rows, "search": "kept" if keep else "rerun", "why": why,
              "selected_before": baseline.get("selected"), "saved_usd": baseline.get("search_cost_usd", 0) if keep else 0,
              "still_verified": sum(r["after"] == "verified" for r in rows),
              "were_verified": sum(r["before"] == "verified" for r in rows)})
        done = baseline.get("round", 1)
        if keep:   # the old proposals stand; the Verifier checks them all again under the new requirement
            return {"results": results, "round": done, "round_base": done - 1, "target": baseline.get("target", False),
                    "reverify": True}
        return {"results": results, "round": done + 1, "round_base": done, "target": False, "reverify": True,
                "feedback": im.warm_start(changes, rows)}

    def after_impact(state: TeamState) -> str:
        kept = baseline and state.get("reverify") and not state.get("feedback")
        return "verifier" if kept else "lead_engineer"

    def lead(state: TeamState) -> dict:
        req, round_ = state["req"], state.get("round", 1) - state.get("round_base", 0)
        if round_ == 1:
            emit({"type": "design_start", "requirement": req.model_dump(), "optimum": _optimum(req),
                  "optimum_by_material": {m: _optimum(req, [m]) for m in MATERIALS}})
            again = ("\n\n" + "\n".join(state["feedback"])) if state.get("feedback") else ""
        else:
            emit({"type": "redesign_start", "round": round_, "directive": state["directive"]})
            again = (f"\n\nThis is design round {round_}. The chief engineer sent the selected design back. "
                     f"Directive: {state['directive']} Write notes that act on it.")
        emit({"type": "agent_start", "agent": "lead"})
        plan = ask("lead", Plan, requirement_text(req) + "\n\nYou lead a team with one specialist agent per material: "
                   f"{', '.join(MATERIALS)}. Each specialist can run the physics check. Before they start, write one "
                   "short note per material: where in the design space to search and why, given the requirements. "
                   "Use the numbers. Keep each note to one or two sentences." + again) or Plan(reasoning="no plan returned", assignments=[])
        notes = {m: next((a.note for a in plan.assignments if a.material == m), "Search the whole envelope.")
                 for m in MATERIALS}  # every material gets a specialist, even if the lead skipped one
        emit({"type": "plan", "agent": "lead", "reasoning": plan.reasoning, "notes": notes, "round": round_})
        emit({"type": "agent_done", "agent": "lead"})
        return {"req": req, "notes": notes}

    def dispatch(state: TeamState) -> list[Send]:
        round_, feedback = state.get("round", 1), state.get("feedback") or []
        if round_ - state.get("round_base", 0) > 1:   # the specialists get the same directive, plus what the review already saw
            feedback = [f"The chief engineer sent the selected design back. Directive: {state['directive']}"]
            feedback += [f"Already verified: {v['design']['material']} {v['design']['width_mm']:g} x "
                         f"{v['design']['thickness_mm']:g} mm at {v['mass_g']:.0f} g" for v in state.get("verified", [])]
        return [Send("material_specialist", {"req": state["req"], "material": m, "note": state["notes"][m], "round": round_,
                                    "feedback": feedback}) for m in MATERIALS]

    def specialist(task: dict) -> dict:
        material, started = task["material"], time.monotonic()
        emit({"type": "agent_start", "agent": material})
        agent = LangGraphProposer(model=model, n=n, max_turns=max_turns, focus=material, note=task["note"],
                                  on_event=emit, chat_model=make_chat())
        try:
            drafts = agent.propose(task["req"], task.get("feedback", []))
        except Exception as e:  # one agent failing must not stop the team
            emit({"type": "agent_error", "agent": material, "error": f"{type(e).__name__}: {e}"})
            return {"results": [{"agent": material, "drafts": [], "usage": agent.usage, "round": task.get("round", 1)}]}
        emit({"type": "agent_done", "agent": material, "proposals": len(drafts),
              "seconds": round(time.monotonic() - started, 1)})
        return {"results": [{"agent": material, "drafts": drafts, "usage": agent.usage, "round": task.get("round", 1)}]}

    def verify_all(state: TeamState) -> dict:
        req, records, round_, again = state["req"], [], state.get("round", 1), state.get("reverify")
        # earlier rounds are already verified, unless the requirement changed under them
        new = [r for r in state["results"] if again or r.get("round", 1) == round_]
        for result in sorted(new, key=lambda r: (r.get("round", 1), list(MATERIALS).index(r["agent"]))):
            made = result.get("round", 1)
            for i, d in enumerate(result["drafts"], 1):
                r = verify(Proposal(**d.model_dump(), id=im.proposal_id(result["agent"], made, i), round=made), req)
                records.append(r)
                c = r.check
                emit({"type": "verified", "agent": result["agent"], "id": r.proposal.id, "round": made,
                      "design": d.design.model_dump(), "label": r.label, "status": r.status,
                      "mass_g": c.mass_g if c else None, "margin_worst": c.worst_case.margin_yield if c else None,
                      "false_pass": any(f.kind == "false_pass" for f in r.findings),
                      "findings": [f.message for f in r.findings], "reason": r.reason})
        # every verified design so far, lightest first, one entry per design
        pool = {}
        for v in ([] if again else state.get("verified", [])) + [{"id": r.proposal.id, "design": r.proposal.design.model_dump(), "mass_g": r.check.mass_g}
                                              for r in records if r.status == "accepted"]:
            pool.setdefault(tuple(v["design"].values()), v)
        verified = sorted(pool.values(), key=lambda v: v["mass_g"])
        if not state.get("target"):
            selected_id, reason = select(records)
            picked = next((r for r in records if r.proposal.id == selected_id), None)
            best = pool[tuple(picked.proposal.design.model_dump().values())] if picked else None
        else:   # after a review, lightest is not enough: the design must also meet the board's target
            best = rv.select_robust(verified, req)
            target = f"P(failure) <= {rv.MAX_P_FAILURE:.0%}"
            if best:
                reason = f"lightest of {len(verified)} verified designs that meets the review target, {target}, at {best['mass_g']:.0f} g"
            else:
                best = verified[0] if verified else None
                reason = (f"no verified design meets the review target, {target}; the lightest verified design stays"
                          if best else "no design passed verification")
        optimum = _optimum(req)
        c = check(Design(**best["design"]), req) if best else None
        emit({"type": "decision", "selected_id": best["id"] if best else None, "reason": reason, "round": round_ - state.get("round_base", 0),
              "selected": None if not best else {"id": best["id"], "design": best["design"], "mass_g": c.mass_g,
                                                 "matrix": [x.model_dump() for x in c.results]},
              "regret": c.mass_g / optimum["mass_g"] - 1 if best and optimum else None,
              "optimum": optimum})
        return {"selected": best["design"] if best else None, "verified": verified, "reverify": False, "feedback": []}

    @memo("redteam", lambda s: {**reads(s), "on": red})
    def redteam(state: TeamState) -> dict:
        if not red or not state.get("selected"):
            return {"attacks": []}
        emit({"type": "agent_start", "agent": "redteam"})
        try:
            report = rt.red_team(Design(**state["selected"]), state["req"], make_chat(), emit, track("redteam"))
        except Exception as e:  # the design stands without the red team, but the gap is shown
            emit({"type": "agent_error", "agent": "redteam", "error": f"{type(e).__name__}: {e}"})
            return {"attacks": []}
        emit({"type": "redteam", **report})
        emit({"type": "agent_done", "agent": "redteam"})
        return {"attacks": report["attacks"]}

    @memo("fragility", lambda s: {**reads(s), "verified": pool_of(s), "qty": qty, "on": monte_carlo})
    def fragility(state: TeamState) -> dict:
        if not monte_carlo or not state.get("selected"):
            return {}
        emit({"type": "agent_start", "agent": "fragility"})
        req = state["req"]
        emit({"type": "fragility", **fr.monte_carlo(Design(**state["selected"]), req)})
        emit({"type": "tradeoffs", "selected_id": next((v["id"] for v in state.get("verified", [])
                                                        if v["design"] == state["selected"]), None),
              "rows": fr.tradeoffs(state.get("verified", [])[:8], req, qty), "scatter": fr.SCATTER})
        emit({"type": "agent_done", "agent": "fragility"})
        return {}

    @memo("chief", lambda s: {**reads(s), "attacks": s.get("attacks"), "agent": chief, "max_rounds": max_rounds,
                              "round": s.get("round", 1) - s.get("round_base", 0)})
    def review_board(state: TeamState) -> dict:
        if not state.get("selected"):
            return {"verdict": "none"}
        req, design = state["req"], Design(**state["selected"])
        round_ = state.get("round", 1) - state.get("round_base", 0)
        emit({"type": "agent_start", "agent": "chief"})
        findings = rv.gate(design, req, state.get("attacks"))
        proposed = None
        if chief:
            try:
                proposed = ask("chief", rv.ReviewDecision,
                               rv.prompt(design, req, findings, round_, max_rounds, check(design, req).mass_g))
            except Exception as e:  # the gate still decides without the agent
                emit({"type": "agent_error", "agent": "chief", "error": f"{type(e).__name__}: {e}"})
        verdict = rv.decide(design, findings, proposed, round_, max_rounds)
        emit({"type": "review", "round": round_, "max_rounds": max_rounds, "design": design.model_dump(),
              "findings": findings, **verdict})
        emit({"type": "agent_done", "agent": "chief"})
        if verdict["decision"] == "redesign":
            return {"verdict": "redesign", "round": state.get("round", 1) + 1, "directive": verdict["directive"],
                    "rejected": [design.model_dump()], "target": True}
        return {"verdict": verdict["decision"]}

    def after_review(state: TeamState) -> str:
        return "lead_engineer" if state.get("verdict") == "redesign" else "compliance_engine"

    @memo("compliance", lambda s: {**reads(s), "verified": pool_of(s), "agent": review})
    def compliance(state: TeamState) -> dict:
        if not state.get("selected"):
            return {}
        req, design = state["req"], Design(**state["selected"])
        emit({"type": "agent_start", "agent": "compliance"})
        items = ce.evaluate(design, req, check(design, req))
        rows = []  # the same rules for every verified design, to compare them
        for v in state.get("verified", [])[:8]:
            d = Design(**v["design"])
            rows.append({"id": v["id"], "mass_g": v["mass_g"], "statuses": [i["status"] for i in ce.evaluate(d, req, check(d, req))]})
        emit({"type": "compliance", "design": design.model_dump(), "items": items, "summary": ce.summary(items),
              "matrix": {"rules": [i["id"] for i in items], "rows": rows}})
        if review:
            try:
                r = ask("compliance", ce.ComplianceReview, ce.review_prompt(design, items))
                if r:
                    emit({"type": "compliance_review", **r.model_dump()})
            except Exception as e:  # the checklist stands without the explanation
                emit({"type": "agent_error", "agent": "compliance", "error": f"{type(e).__name__}: {e}"})
        emit({"type": "agent_done", "agent": "compliance"})
        return {"needs": ce.supplier_requirements(design)}

    def rfq(state: TeamState):
        if not suppliers or not state.get("selected") or state.get("verdict") == "escalate":
            return "procurement"   # nothing is bought before the design is released
        design = Design(**state["selected"])
        emit({"type": "rfq", "qty": qty, "needs": state["needs"], "text": sd.rfq_text(design, state["req"], qty, state["needs"]),
              "vendors": [{"id": v["id"], "name": v["name"]} for v in sd.VENDORS]})
        return [Send("supplier_agent", {"req": state["req"], "design": state["selected"], "needs": state["needs"], "vendor": v})
                for v in sd.VENDORS]

    @memo(lambda t: "vendor-" + t["vendor"]["id"],
          lambda t: {"design": t["design"], "req.length_mm": t["req"].length_mm, "needs": t["needs"], "qty": qty, "vendor": t["vendor"]})
    def vendor(task: dict) -> dict:
        v, agent = task["vendor"], "vendor-" + task["vendor"]["id"]
        emit({"type": "agent_start", "agent": agent})
        try:
            result = sd.ask_vendor(v, Design(**task["design"]), task["req"], qty, task["needs"], make_chat(), emit, track(agent))
        except Exception as e:
            emit({"type": "agent_error", "agent": agent, "error": f"{type(e).__name__}: {e}"})
            return {"quotes": []}
        emit({"type": "quote", **result})
        emit({"type": "agent_done", "agent": agent})
        return {"quotes": [result]}

    @memo("procurement", lambda s: {"design": s.get("selected"), "qty": qty,
                                    "quotes": sorted(s.get("quotes", []), key=lambda q: q["vendor"])})
    def procurement(state: TeamState) -> dict:
        quotes = sorted(state.get("quotes", []), key=lambda q: q["vendor"])
        if not quotes:
            return {}
        emit({"type": "agent_start", "agent": "procurement"})
        ranked = sd.rank(quotes, qty)
        memo = None
        try:
            m = ask("procurement", Memo, sd.memo_prompt(Design(**state["selected"]), quotes, ranked, qty))
            memo = m.memo if m else None
        except Exception as e:
            emit({"type": "agent_error", "agent": "procurement", "error": f"{type(e).__name__}: {e}"})
        emit({"type": "procurement", "qty": qty, "weights": sd.WEIGHTS, "ranking": ranked,
              "selected": ranked[0]["vendor"] if ranked else None, "memo": memo})
        emit({"type": "agent_done", "agent": "procurement"})
        return {}

    @memo("market", lambda s: {**reads(s, "length_mm"), "on": market, "held": s.get("verdict") == "escalate"})
    def scan(state: TeamState) -> dict:
        if not market or not state.get("selected") or state.get("verdict") == "escalate":
            return {}
        emit({"type": "agent_start", "agent": "market"})
        try:
            found = find_vendors(Design(**state["selected"]), state["req"], model, client=vendor_client)
        except Exception as e:  # no vendors is not a failed design
            emit({"type": "agent_error", "agent": "market", "error": f"{type(e).__name__}: {e}"})
            return {}
        emit({"type": "usage", "agent": "market", **found["usage"]})
        emit({"type": "market", **{k: v for k, v in found.items() if k != "usage"}})
        emit({"type": "agent_done", "agent": "market"})
        return {}

    graph = StateGraph(TeamState, input_schema=TeamInput)
    graph.add_node("requirements_desk", requirements, retry_policy=RETRY)
    graph.add_node("human_approval", approve)
    graph.add_node("change_impact", impact)
    graph.add_node("lead_engineer", lead, retry_policy=RETRY)
    graph.add_node("material_specialist", specialist)
    graph.add_node("verifier", verify_all)
    graph.add_node("red_team_review", redteam)
    graph.add_node("monte_carlo", fragility)
    graph.add_node("chief_engineer", review_board)
    graph.add_node("compliance_engine", compliance)
    graph.add_node("supplier_agent", vendor)
    graph.add_node("procurement", procurement)
    graph.add_node("market_scan", scan)
    graph.add_edge(START, "requirements_desk")
    graph.add_edge("requirements_desk", "human_approval")
    graph.add_edge("human_approval", "change_impact")
    graph.add_conditional_edges("change_impact", after_impact, ["lead_engineer", "verifier"])   # a change run can keep the old search
    graph.add_conditional_edges("lead_engineer", dispatch, ["material_specialist"])
    graph.add_edge("material_specialist", "verifier")
    graph.add_edge("verifier", "red_team_review")
    graph.add_edge("red_team_review", "monte_carlo")
    graph.add_edge("monte_carlo", "chief_engineer")
    graph.add_conditional_edges("chief_engineer", after_review, ["lead_engineer", "compliance_engine"])   # the loop: back to the lead, or on
    graph.add_conditional_edges("compliance_engine", rfq, ["supplier_agent", "procurement"])
    graph.add_edge("supplier_agent", "procurement")
    graph.add_edge("procurement", "market_scan")
    graph.add_edge("market_scan", END)
    return graph.compile(checkpointer=checkpointer), usage


def run_team(req: Requirement | None, model: str, emit, n: int = 4, make_chat=None, market: bool = True,
             vendor_client=None, suppliers: bool = True, review: bool = True, red: bool = True,
             brief: str | None = None, approve=None, chief: bool = True, max_rounds: int = rv.MAX_ROUNDS,
             qty: int = 10, baseline: dict | None = None) -> dict:
    """Run the whole team once. Every event goes to emit(), with t = seconds since the start.

    Returns a snapshot of the run. Pass it back as baseline, with a changed req or qty, and only the
    steps that read the change run again (impact.py).

    With a brief, the Requirements Desk runs first. approve(payload) -> {"edits": {...}} is called when the
    graph stops for approval; without it the requirement is approved as drafted.
    """
    start = time.monotonic()

    chain = ledger.Chain(emit)   # numbers every event and chains it to the one before by a hash

    def timed(event: dict) -> None:
        chain({**event, "t": round(time.monotonic() - start, 2)})

    timed({"type": "run_start", "case": req.case if req else "brief", "model": model, "materials": list(MATERIALS),
           "requirement": req.model_dump() if req else None, "brief": brief,
           "optimum": _optimum(req) if req else None, "change": bool(baseline), "code_version": ledger.code_version()})
    approval = bool(brief)
    steps = im.Steps(baseline)
    graph, usage = build_team(model, n, timed, make_chat, market=market, vendor_client=vendor_client, suppliers=suppliers,
                              review=review, red=red, approval=approval, chief=chief, max_rounds=max_rounds, qty=qty,
                              checkpointer=MemorySaver() if approval else None, baseline=baseline, steps=steps)
    config = {"configurable": {"thread_id": f"run-{start}"}, "recursion_limit": 25 + 10 * max_rounds,
              "max_concurrency": MAX_CONCURRENCY}
    state = {"brief": brief} if brief else {"req": req}
    result = graph.invoke(state, config)
    while result.get("__interrupt__"):   # the graph stopped for a person; resume it with their answer
        payload = result["__interrupt__"][0].value
        timed({"type": "approval_needed", **payload})
        decision = approve(payload) if approve else {"edits": {}}
        result = graph.invoke(Command(resume=decision), config)
    timed({"type": "run_end", "seconds": round(time.monotonic() - start, 1)})
    if "lead" in usage or not baseline:
        lead_cost = (usage["lead"].cost_usd or 0) if "lead" in usage else 0
    else:   # the search was kept, so its cost is the baseline's
        lead_cost = baseline.get("search_cost_usd", 0) - sum(r.get("cost_usd") or 0 for r in baseline["results"])
    return im.snapshot(result, steps, qty, lead_cost)
