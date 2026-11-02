"""The loop: propose, verify, feed the results back, repeat. Then pick from verified designs only."""
import time
from datetime import datetime, timezone

from .schemas import DecisionTrace, Proposal, Record, Requirement
from .verify import feedback, select, verify


def run(req: Requirement, proposer, rounds: int = 1) -> DecisionTrace:
    started = time.monotonic()
    records: list[Record] = []
    notes: list[str] = []
    for rnd in range(1, rounds + 1):
        drafts = proposer.propose(req, notes)
        batch = [verify(Proposal(**d.model_dump(), id=f"r{rnd}-{i + 1:02d}", round=rnd), req)
                 for i, d in enumerate(drafts)]
        records += batch
        notes = feedback(batch)

    selected_id, reason = select(records)
    return DecisionTrace(
        requirement=req,
        proposer=proposer.name,
        model=proposer.model,
        rounds=rounds,
        records=records,
        selected_id=selected_id,
        selection_reason=reason,
        usage=proposer.usage,
        seconds=round(time.monotonic() - started, 1),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def summarize(trace: DecisionTrace) -> dict:
    findings = [f for r in trace.records for f in r.findings]
    labels = [r.label for r in trace.records]
    return {
        "proposals": len(trace.records),
        "verified": labels.count("verified"),
        "fails": labels.count("fails"),
        "fragile": labels.count("fragile"),
        "not_manufacturable": labels.count("not manufacturable"),
        "out_of_envelope": labels.count("out of envelope"),
        "false_passes_caught": sum(f.kind == "false_pass" for f in findings),
        "claims_off": sum(f.kind == "claim_off" for f in findings),
    }
