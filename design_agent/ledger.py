"""Evidence ledger: every event of a run is chained to the one before it by a hash.

    h(n) = sha256( h(n-1) + the event as sorted JSON )

Change, drop or reorder one recorded event and every hash after it stops matching, so verify()
can say where a recording was altered. This makes a recording tamper-evident, not tamper-proof:
someone who rewrites the whole file can rewrite the whole chain. The last hash, the head, is
printed on the Design Package for that reason. Keep it somewhere else and the recording can be
checked against it later.

entries() then turns the events into the rows a reviewer reads: who decided what, whether that
was an agent, plain code or a person, and which event is the evidence.
"""
import hashlib
import json
import subprocess
import threading
from pathlib import Path

GENESIS = "0" * 16


def _digest(previous: str, event: dict) -> str:
    body = json.dumps({k: v for k, v in event.items() if k != "h"}, sort_keys=True, default=str)
    return hashlib.sha256((previous + body).encode()).hexdigest()[:16]


class Chain:
    """Numbers and chains events as they are sent. Parallel agents send at once, so one at a time."""

    def __init__(self, emit):
        self.emit, self.head, self.seq, self.lock = emit, GENESIS, 0, threading.Lock()

    def __call__(self, event: dict) -> None:
        with self.lock:   # the chain order and the order in the recording must be the same
            self.seq += 1
            event = json.loads(json.dumps({**event, "seq": self.seq}, default=str))   # exactly what the file will hold
            self.head = event["h"] = _digest(self.head, event)
            self.emit(event)


def verify(events: list[dict]) -> dict:
    """Recompute the chain of a recording. Events without a hash (older recordings) are counted, not checked."""
    chained = [e for e in events if "h" in e]
    head, broken = GENESIS, None
    for n, e in enumerate(chained, 1):
        if e.get("seq") != n or _digest(head, e) != e["h"]:
            broken = n
            break
        head = e["h"]
    return {"events": len(events), "chained": len(chained), "ok": bool(chained) and broken is None,
            "broken_at": broken, "head": head if chained and broken is None else None}


def code_version() -> str:
    """The commit the run was made with, and whether the working tree had changes on top of it."""
    try:
        def git(*args):
            return subprocess.run(["git", *args], cwd=Path(__file__).parent, capture_output=True, text=True, timeout=5).stdout.strip()
        commit = git("rev-parse", "--short", "HEAD")
        return (commit + ("+changes" if git("status", "--porcelain") else "")) if commit else "unknown"
    except Exception:
        return "unknown"


def _design(d: dict | None) -> str:
    return f"{d['material']} {d['width_mm']:g} x {d['thickness_mm']:g} mm" if d else "no design"


# event type -> (who, kind, what it says). kind: agent, code or person.
def _row(e: dict) -> tuple[str, str, str] | None:
    t = e["type"]
    if t == "run_start":
        return "Runner", "code", f"run started: {e.get('case')}, model {e.get('model')}, code {e.get('code_version', 'unknown')}"
    if t == "requirements" and e.get("mode") == "brief":
        return "Requirements Desk", "agent + code", (f"drafted the requirement from the brief; {len(e['draft']['sources'])} sourced values, "
                                                     f"{len(e['derived'])} computed by code, {len(e['assumed'])} assumed")
    if t == "approved":
        edits = ", ".join(f"{k} = {v}" for k, v in (e.get("edits") or {}).items())
        return "Person", "person", "approved the requirement" + (f" with edits: {edits}" if edits else " as drafted")
    if t == "impact":
        what = ", ".join(f"{c['field']} {c['old']:g} -> {c['new']:g}" for c in e["changes"]) or "nothing"
        return "Change impact", "code", f"changed: {what}; {e['still_verified']} of {e['were_verified']} old designs still verified; search {e['search']}"
    if t == "step":
        why = "inputs unchanged" if e["action"] == "reused" else "changed: " + ", ".join(e.get("changed", []))
        return "Change impact", "code", f"step {e['step']} {'reused' if e['action'] == 'reused' else 'ran again'} ({why})"
    if t == "plan":
        return "Lead", "agent", "briefed the specialists: " + e["reasoning"][:160]
    if t == "verified":
        extra = "; FALSE PASS caught" if e.get("false_pass") else ""
        return "Verifier", "code", f"{e['id']} {_design(e['design'])}: {e['label']}{extra}"
    if t == "decision":
        return "Verifier", "code", (f"selected {e['selected_id']} {_design(e['selected']['design'])}: {e['reason']}" if e.get("selected")
                                    else f"no design selected: {e['reason']}")
    if t == "redteam":
        n = lambda k: sum(a["result"] == k for a in e["attacks"])
        return "Red Team", "agent + code", (f"{len(e['attacks'])} attacks on {_design(e['design'])}: {n('confirmed')} confirmed by code, "
                                            f"{n('refuted')} refuted, {n('needs analysis')} not testable here")
    if t == "fragility":
        return "Monte Carlo", "code", f"{_design(e['design'])}: {e['failures']} of {e['samples']} samples fail, P(failure) {e['p_failure']:.2%}"
    if t == "review":
        over = f"; the agent proposed {e['proposed']}, overruled: {e['rule']}" if e.get("overruled") else ""
        return "Chief Engineer", "agent + code", f"round {e['round']}: {e['decision']} for {_design(e['design'])}{over}"
    if t == "compliance":
        s = e["summary"]
        return "Compliance Engine", "code", f"{s['pass']} rules pass, {s['action']} need evidence or action, {s['fail']} fail"
    if t == "compliance_review":
        return "Compliance review", "agent", f"explained the gaps and proposed {len(e['actions'])} actions; changed no status"
    if t == "quote":
        q = e["quote"]
        said = f"bid ${q['price_each_usd']:.2f} each" if q["bid"] and q["price_each_usd"] else "no bid"
        checks = "; ".join(e["flags"] + ([] if e["eligible"] else ["not eligible: " + "; ".join(e["blockers"])])) or "checks pass"
        return e["name"], "agent + code", f"{said}; code: {checks}"
    if t == "procurement":
        return "Procurement", "code + agent", (f"code ranked {len(e['ranking'])} valid bids, first: vendor {e['selected']}; an agent wrote the memo"
                                               if e["ranking"] else "no valid bid to rank")
    if t == "market":
        return "Market scan", "agent + code", f"{len(e['vendors'])} shops kept, {e.get('dropped', 0)} dropped because their link was not in the search results"
    if t == "agent_error":
        return str(e.get("agent")), "agent", "FAILED: " + str(e.get("error"))[:160]
    if t == "run_end":
        return "Runner", "code", f"run ended after {e['seconds']} s"
    return None


def entries(events: list[dict]) -> list[dict]:
    """The decision record: one row per event that decided, checked or approved something."""
    rows = []
    for e in events:
        row = _row(e)
        if row:
            rows.append({"seq": e.get("seq"), "t": e.get("t"), "who": row[0], "kind": row[1], "what": row[2],
                         "hash": e.get("h"), "reused": bool(e.get("reused"))})
    return rows
