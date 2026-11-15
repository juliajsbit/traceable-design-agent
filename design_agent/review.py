"""Chief Engineer review: the gate between analysis and release, and the way back to the design team.

The Verifier only proves a design meets the written requirements. The Red Team and the Monte Carlo
then learn things about the selected design that no requirement states, for example that it fails
in 2% of the sampled parts. This review decides what to do with that:

    ship      release the design to compliance and procurement
    redesign  send it back to the lead with a directive; the team searches again
    escalate  no rounds left, or no design can meet the target: stop and ask a person

Same split as everywhere else here. Code computes the findings and marks which ones block a
release. An agent reads them and proposes the decision and the directive. Code then holds the
agent to three rules (decide below): it cannot ship past a blocking finding, it cannot ask for a
redesign without one, and it cannot loop past the round limit.

The limits are illustrative program policy, not a standard.
"""
from typing import Literal

from pydantic import BaseModel

from .fragility import monte_carlo
from .proposers import requirement_text
from .redteam import breaking_point
from .schemas import Design, Requirement

MAX_P_FAILURE = 0.01   # the board releases a design only if at most 1% of sampled parts fail
MAX_ROUNDS = 2         # one redesign, then a person decides


class ReviewDecision(BaseModel):
    decision: Literal["ship", "redesign", "escalate"]
    rationale: str
    directive: str = ""          # for a redesign: what the team should change, in numbers
    conditions: list[str] = []   # for a release: what is still open and who should close it


def robustness(design: Design, req: Requirement) -> dict:
    """What the board looks at for one design. Plain code, the same seed every time."""
    mc, bp = monte_carlo(design, req), breaking_point(design, req)
    return {"p_failure": mc["p_failure"], "p5": mc["p5"], "load_reserve": bp["load_multiplier"],
            "thickness_reserve_mm": bp["thickness_loss_mm"]}


def meets_targets(design: Design, req: Requirement) -> bool:
    return monte_carlo(design, req)["p_failure"] <= MAX_P_FAILURE


def select_robust(verified: list[dict], req: Requirement) -> dict | None:
    """The lightest verified design that also meets the board's target, or None."""
    for v in sorted(verified, key=lambda v: v["mass_g"]):
        if meets_targets(Design(**v["design"]), req):
            return v
    return None


def finding(id_, source, title, value, limit, blocking, detail=""):
    return {"id": id_, "source": source, "title": title, "value": value, "limit": limit, "blocking": blocking, "detail": detail}


def gate(design: Design, req: Requirement, attacks: list[dict] | None = None) -> list[dict]:
    """Findings on the selected design. Only code sets `blocking`."""
    r = robustness(design, req)
    load = f"x{r['load_reserve']:g}" if r["load_reserve"] is not None else "> x20"
    thin = f"-{r['thickness_reserve_mm']:g} mm" if r["thickness_reserve_mm"] is not None else "any"
    confirmed = [a for a in attacks or [] if a.get("result") == "confirmed"]
    untested = [a for a in attacks or [] if a.get("result") == "needs analysis"]
    return [
        finding("G-01", "Monte Carlo", "Probability of failure under scatter", f"{r['p_failure']:.2%}",
                f"<= {MAX_P_FAILURE:.0%}", r["p_failure"] > MAX_P_FAILURE,
                f"5% of sampled parts have a governing margin below {r['p5']:+.3f}."),
        finding("G-02", "Breaking point", "Load the part survives", load, "shown, no limit", False,
                f"Requirements already cover +{req.load_uncertainty:.0%}; this is what is left above that."),
        finding("G-03", "Breaking point", "Plate thinning the part survives", thin, "shown, no limit", False,
                f"Requirements already cover -{req.thickness_tolerance_mm:g} mm."),
        finding("G-04", "Red Team", "Attacks confirmed by code", str(len(confirmed)), "shown, no limit", False,
                "; ".join(a["title"] for a in confirmed) or "none"),
        finding("G-05", "Red Team", "Risks no tool here can test", str(len(untested)), "carried as conditions", False,
                "; ".join(a["title"] for a in untested) or "none"),
    ]


def prompt(design: Design, req: Requirement, findings: list[dict], round_: int, max_rounds: int, mass_g: float) -> str:
    rows = "\n".join(f"{f['id']} [{'BLOCKING' if f['blocking'] else 'ok'}] {f['source']}: {f['title']} = {f['value']} "
                     f"(limit {f['limit']}). {f['detail']}" for f in findings)
    return (f"{requirement_text(req)}\n\nYou are the chief engineer. The team selected {design.material}, "
            f"{design.width_mm:g} x {design.thickness_mm:g} mm, {mass_g:.0f} g. It passed verification. "
            f"This is review round {round_} of {max_rounds}. Findings from the red team and the Monte Carlo:\n\n{rows}\n\n"
            "Decide: ship, redesign or escalate. Use only these findings and cite their ids. A blocking finding "
            "cannot ship. For a redesign, write a directive the design team can act on with its physics check: "
            "say what to change and give a number to aim for, for example a worst-case yield margin. Escalate when "
            "another round cannot fix it, for example when the envelope leaves no room. For a release, list what "
            "is still open as conditions, each with an owner. Keep the rationale to two sentences.")


def default_directive(design: Design, blocking: list[dict]) -> str:
    why = "; ".join(f"{f['id']} {f['title'].lower()} is {f['value']}, limit {f['limit']}" for f in blocking)
    return (f"{design.material} {design.width_mm:g} x {design.thickness_mm:g} mm was sent back: {why}. "
            "Find a design with more worst-case margin, a wider or thicker section or a stronger material.")


def decide(design: Design, findings: list[dict], proposed: ReviewDecision | None, round_: int,
           max_rounds: int = MAX_ROUNDS) -> dict:
    """The final decision. The agent proposes; these rules have the last word."""
    blocking = [f for f in findings if f["blocking"]]
    asked = proposed.decision if proposed else None
    if not blocking:
        decision, rule = "ship", "a redesign needs a blocking finding" if asked not in (None, "ship") else None
    elif round_ >= max_rounds:
        decision, rule = "escalate", "no rounds left" if asked != "escalate" else None
    elif asked == "escalate":
        decision, rule = "escalate", None
    else:
        decision, rule = "redesign", "a blocking finding cannot ship" if asked == "ship" else None
    directive = ""
    if decision == "redesign":
        directive = (proposed.directive if proposed and asked == "redesign" and proposed.directive.strip()
                     else default_directive(design, blocking))
    return {"decision": decision, "proposed": asked, "overruled": rule is not None, "rule": rule,
            "rationale": proposed.rationale if proposed else "", "directive": directive,
            "conditions": proposed.conditions if proposed and decision == "ship" else [],
            "blocking": [f["id"] for f in blocking]}
