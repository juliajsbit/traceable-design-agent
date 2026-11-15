"""Red Team Review: an agent tries to break the selected design, and code checks every attack it claims.

The agent thinks of ways the real world could differ from the analysis (heavier payload, harder
launch, a thinner plate after machining, a longer arm after a layout change) and tests them with
its stress_test tool. It may also name risks no tool here can test, such as fatigue. Code then
re-runs every testable attack itself and computes the breaking point, so a wrong claim from the
red team is caught the same way a wrong claim from a designer is.
"""
import json
from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from .physics import analyze
from .proposers import requirement_text
from .schemas import Design, Requirement


class Attack(BaseModel):
    title: str
    rationale: str
    testable: bool
    load_multiplier: float = 1.0
    thickness_loss_mm: float = 0.0
    width_loss_mm: float = 0.0
    length_increase_mm: float = 0.0
    expected_to_fail: bool
    severity: Literal["low", "medium", "high"]


class RedTeamReport(BaseModel):
    attacks: list[Attack]


def survives(design: Design, req: Requirement, load_multiplier=1.0, thickness_loss_mm=0.0, width_loss_mm=0.0,
             length_increase_mm=0.0) -> dict:
    """Strength and stiffness under changed conditions, at nominal (no extra safety tolerance on top)."""
    d = design.model_copy(update={"width_mm": max(design.width_mm - width_loss_mm, 0.1)})
    r = req.model_copy(update={"length_mm": req.length_mm + length_increase_mm})
    c = analyze(d, r, req.load_n * load_multiplier, max(design.thickness_mm - thickness_loss_mm, 0.1))
    ok = c.margin_yield >= 0 and c.margin_ultimate >= 0 and c.deflection_mm <= req.max_deflection_mm
    return {"survives": ok, "margin_yield": round(c.margin_yield, 3), "margin_ultimate": round(c.margin_ultimate, 3),
            "deflection_mm": round(c.deflection_mm, 3), "deflection_limit_mm": req.max_deflection_mm}


def breaking_point(design: Design, req: Requirement) -> dict:
    """How much more load, and how much thinner a plate, before the part stops meeting strength or stiffness."""
    def bisect(test, hi):
        lo = 0.0
        if test(hi):
            return None   # still survives at the top of the range
        for _ in range(40):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if test(mid) else (lo, mid)
        return lo
    load = bisect(lambda m: survives(design, req, load_multiplier=m)["survives"], 20.0)
    thin = bisect(lambda t: survives(design, req, thickness_loss_mm=t)["survives"], design.thickness_mm - 0.1)
    return {"load_multiplier": round(load, 2) if load is not None else None,
            "thickness_loss_mm": round(thin, 2) if thin is not None else None}


def prompt(design: Design, req: Requirement) -> str:
    return (f"{requirement_text(req)}\n\nThe team selected: {design.material}, {design.width_mm:g} x {design.thickness_mm:g} mm. "
            "You are the red team. Your job is to break this design before it flies. Think of realistic ways the "
            "real world could differ from the analysis: a heavier payload, a harder launch, a thinner plate after "
            "machining, a longer arm after a layout change. Test them with stress_test. Also name risks the tool "
            "cannot test, such as fatigue, thermal cycling or corrosion, and mark them not testable. "
            "Then report five attacks.")


def red_team(design: Design, req: Requirement, llm, emit=lambda e: None, track=lambda m: None) -> dict:
    @tool
    def stress_test(load_multiplier: float = 1.0, thickness_loss_mm: float = 0.0, width_loss_mm: float = 0.0,
                    length_increase_mm: float = 0.0) -> str:
        """Re-run strength and stiffness for the selected design with more load, a thinner or narrower plate, or a longer arm."""
        return json.dumps(survives(design, req, load_multiplier, thickness_loss_mm, width_loss_mm, length_increase_mm))

    agent = llm.bind_tools([stress_test])
    messages = [HumanMessage(prompt(design, req))]
    for _ in range(6):
        reply = agent.invoke(messages)
        track(reply)
        messages.append(reply)
        emit({"type": "redteam_step", "tests": [c["args"] for c in reply.tool_calls]})
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            messages.append(stress_test.invoke(call))
    messages.append(HumanMessage("Report your five attacks now."))
    report = None
    for _ in range(2):  # history has tool calls, so the schema goes through a tool; retry once if malformed
        result = llm.with_structured_output(RedTeamReport, include_raw=True).invoke(messages)
        track(result["raw"])
        if result["parsed"]:
            report = result["parsed"]
            break
    attacks = []
    for a in (report.attacks if report else []):
        if not a.testable:
            attacks.append({**a.model_dump(), "result": "needs analysis", "agent_right": None})
            continue
        r = survives(design, req, a.load_multiplier, a.thickness_loss_mm, a.width_loss_mm, a.length_increase_mm)
        attacks.append({**a.model_dump(), **r, "result": "refuted" if r["survives"] else "confirmed",
                        "agent_right": a.expected_to_fail == (not r["survives"])})
    return {"design": design.model_dump(), "attacks": attacks, "breaking_point": breaking_point(design, req)}
