"""Change impact: a requirement changes after a run, and only the steps that read it run again.

It works like a build system. Every step of the team declares its inputs (the selected design,
the requirement fields it reads, the quotes, ...). The inputs are hashed. A finished run saves,
for each step, that hash together with the events the step sent and the state it returned. A
change run starts from such a saved run, the baseline:

    1. diff      which requirement fields changed, and is each change tighter or looser
    2. reverify  code checks every design of the baseline again under the new requirement;
                 this costs nothing and shows what the change broke
    3. search    the design search is kept when no agent could do better: every change is tighter
                 and the selected design still passes. Otherwise the specialists search again,
                 starting from what the reverification found
    4. steps     each later step compares its input hash with the baseline. Same hash: its saved
                 events are sent again, marked reused, with no model call. Different hash: it runs,
                 and the event names the inputs that changed

So the answer to "why did this step run again" is always a list of inputs, never a guess.
"""
import hashlib
import json
from pathlib import Path

from .physics import check
from .schemas import Design, Proposal, ProposalDraft, Requirement
from .verify import verify

# +1: a larger value is harder to meet. -1: a smaller value is harder to meet.
HARDER = {"load_n": +1, "length_mm": +1, "yield_factor": +1, "ultimate_factor": +1,
          "max_deflection_mm": -1, "width_max_mm": -1, "thickness_max_mm": -1}
UNIT = {"load_n": "N", "length_mm": "mm", "max_deflection_mm": "mm", "width_max_mm": "mm", "thickness_max_mm": "mm"}
SHORT = {"Al 6061-T6": "Al", "Ti-6Al-4V": "Ti", "Steel 4130": "Steel"}


def editable(req: Requirement) -> dict:
    """The values a person can change, by the same names the approval step uses."""
    return {"load_n": req.load_n, "length_mm": req.length_mm, "yield_factor": req.yield_factor,
            "ultimate_factor": req.ultimate_factor, "max_deflection_mm": req.max_deflection_mm,
            "width_max_mm": req.width_range_mm[1], "thickness_max_mm": req.thickness_range_mm[1]}


def apply_edits(req: Requirement, edits: dict) -> Requirement:
    update = {k: float(v) for k, v in edits.items() if k in ("load_n", "length_mm", "max_deflection_mm", "yield_factor", "ultimate_factor")}
    if "width_max_mm" in edits:
        update["width_range_mm"] = (req.width_range_mm[0], float(edits["width_max_mm"]))
    if "thickness_max_mm" in edits:
        update["thickness_range_mm"] = (req.thickness_range_mm[0], float(edits["thickness_max_mm"]))
    new = req.model_copy(update=update)
    if (new.width_range_mm[1] <= new.width_range_mm[0] or new.thickness_range_mm[1] <= new.thickness_range_mm[0]
            or min(new.load_n, new.length_mm, new.max_deflection_mm, new.yield_factor, new.ultimate_factor) <= 0):
        raise ValueError("the edited requirement does not make sense")
    return new


def diff(old: Requirement, new: Requirement) -> list[dict]:
    a, b = editable(old), editable(new)
    return [{"field": k, "old": a[k], "new": b[k], "unit": UNIT.get(k, ""),
             "effect": "tighter" if (b[k] - a[k]) * HARDER[k] > 0 else "looser"} for k in a if a[k] != b[k]]


def proposal_id(agent: str, round_: int, i: int) -> str:
    return f"{SHORT[agent]}{'' if round_ == 1 else f'-r{round_}'}-{i}"


def reverify(results: list[dict], old: Requirement, new: Requirement) -> list[dict]:
    """Every proposal of the baseline, checked under the old and the new requirement."""
    rows = []
    for result in results:
        for i, d in enumerate(result["drafts"], 1):
            id_ = proposal_id(result["agent"], result.get("round", 1), i)
            p = Proposal(**d.model_dump(), id=id_, round=result.get("round", 1))
            before, after = verify(p, old), verify(p, new)
            rows.append({"id": id_, "design": d.design.model_dump(), "before": before.label, "after": after.label,
                         "mass_g": after.check.mass_g if after.check else None,
                         "reason": after.reason if after.label != "verified" else ""})
    return rows


def keep_search(changes: list[dict], selected: dict | None, new: Requirement) -> tuple[bool, str]:
    """Can the baseline's design search stand, or do the specialists have to search again?"""
    if not selected:
        return False, "the baseline had no design, so the team searches again"
    if not changes:
        return True, "the requirement did not change, so the search stands"
    looser = [c["field"] for c in changes if c["effect"] == "looser"]
    if looser:
        return False, f"{', '.join(looser)} got looser, so a lighter design may now exist: the team searches again"
    d = Design(**selected)
    if not check(d, new).passes:
        return False, (f"the selected {d.material} {d.width_mm:g} x {d.thickness_mm:g} mm no longer passes, "
                       "so the team searches again, starting from the designs that still do")
    return True, ("every change is tighter and the selected design still passes, so no lighter design can have "
                  "appeared: the search stands and only the checks run again")


def warm_start(changes: list[dict], rows: list[dict]) -> list[str]:
    """What the specialists are told before they search again."""
    lines = ["The requirement changed after your previous search: " +
             ", ".join(f"{c['field']} {c['old']:g} -> {c['new']:g}" for c in changes) + ". Your earlier designs, checked again:"]
    for r in rows:
        d = r["design"]
        head = f"{d['material']} {d['width_mm']:g} x {d['thickness_mm']:g} mm"
        lines.append(f"{head} still passes at {r['mass_g']:.0f} g" if r["after"] == "verified" else f"{head} now {r['after']}: {r['reason']}")
    return lines


def canonical(x):
    """The same value must hash the same before and after it is saved as JSON: 10 and 10.0, a tuple and a list."""
    if isinstance(x, dict):
        return {str(k): canonical(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [canonical(v) for v in x]
    return float(x) if isinstance(x, int) and not isinstance(x, bool) else x


def fingerprint(inputs: dict) -> str:
    return hashlib.sha256(json.dumps(canonical(inputs), sort_keys=True, default=str).encode()).hexdigest()[:16]


class Steps:
    """The step cache of one run: looks steps up in the baseline, collects what this run did."""

    def __init__(self, baseline: dict | None = None):
        self.base = (baseline or {}).get("steps", {})
        self.done: dict[str, dict] = {}

    def lookup(self, step: str, inputs: dict) -> tuple[dict | None, list[str]]:
        """(the saved step, []) when the inputs match the baseline, else (None, the inputs that changed)."""
        saved = self.base.get(step)
        if saved is None:
            return None, ["not in the baseline"]
        if saved["key"] == fingerprint(inputs):
            self.done[step] = saved
            return saved, []
        was = saved["inputs"]
        return None, sorted(k for k in inputs.keys() | was.keys() if fingerprint({"v": inputs.get(k)}) != was.get(k))

    def store(self, step: str, inputs: dict, events: list[dict], update: dict) -> None:
        if any(e["type"] == "agent_error" for e in events):
            return   # a failed step is never reused; it runs again next time
        cost = {e["agent"]: e.get("cost_usd") or 0 for e in events if e["type"] == "usage"}   # usage is a running total per agent
        self.done[step] = {"key": fingerprint(inputs), "inputs": {k: fingerprint({"v": v}) for k, v in inputs.items()},
                           "events": events, "update": update, "cost_usd": round(sum(cost.values()), 5)}


def snapshot(state: dict, steps: Steps, qty: int, lead_cost: float) -> dict:
    """Everything a later change run needs from this one. Plain JSON."""
    results = [{"agent": r["agent"], "round": r.get("round", 1), "drafts": [d.model_dump() for d in r["drafts"]],
                "cost_usd": r["usage"].cost_usd if r.get("usage") is not None else r.get("cost_usd", 0)}
               for r in state.get("results", [])]
    return {"req": state["req"].model_dump(), "qty": qty, "round": state.get("round", 1), "target": bool(state.get("target")),
            "selected": state.get("selected"), "verdict": state.get("verdict"), "results": results,
            "search_cost_usd": round(lead_cost + sum(r["cost_usd"] or 0 for r in results), 5), "steps": steps.done}


def save(folder: Path, run_id: str, snap: dict, base: str | None = None) -> None:
    (folder / f"{run_id}.snapshot.json").write_text(json.dumps({"run_id": run_id, "base": base, **snap}))


def load(folder: Path, run_id: str) -> dict | None:
    path = (folder / f"{run_id}.snapshot.json").resolve()
    if path.parent != folder.resolve() or not path.exists():
        return None
    return json.loads(path.read_text())


def restore_results(baseline: dict) -> list[dict]:
    return [{**r, "drafts": [ProposalDraft(**d) for d in r["drafts"]], "usage": None} for r in baseline["results"]]
