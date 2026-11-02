"""Is the AI worth it? Run every proposer on every case and compare against the true optimum."""
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from .cases import oracle
from .pipeline import run, summarize
from .schemas import DecisionTrace, Requirement


class CaseResult(BaseModel):
    case: str
    proposer: str
    trace: DecisionTrace | None = None
    error: str | None = None   # a crash is recorded, never scored as a bad design
    optimum_g: float | None = None
    best_g: float | None = None
    regret: float | None = None  # how much heavier than the optimum; 0 means it found the optimum


def score(req: Requirement, name: str, trace: DecisionTrace) -> CaseResult:
    best = oracle(req)
    optimum = best[1] if best else None
    selected = next((r for r in trace.records if r.proposal.id == trace.selected_id), None)
    best_g = selected.check.mass_g if selected else None
    regret = (best_g / optimum - 1) if best_g and optimum else None
    return CaseResult(case=req.case, proposer=name, trace=trace, optimum_g=optimum, best_g=best_g, regret=regret)


def run_case(req: Requirement, name: str, make_proposer) -> CaseResult:
    try:
        trace = run(req, make_proposer(), rounds=1)
    except Exception as e:
        best = oracle(req)
        return CaseResult(case=req.case, proposer=name, error=f"{type(e).__name__}: {e}", optimum_g=best[1] if best else None)
    return score(req, name, trace)


def rescore(results: list[CaseResult]) -> list[CaseResult]:
    """Score saved traces again, for example after the oracle changes. No model calls."""
    return [score(r.trace.requirement, r.proposer, r.trace) if r.trace else r for r in results]


def run_eval(cases: dict[str, Requirement], proposers: dict, concurrency: int = 4) -> list[CaseResult]:
    jobs = [(req, name, make) for name, make in proposers.items() for req in cases.values()]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda job: run_case(*job), jobs))


def scoreboard(results: list[CaseResult]) -> list[dict]:
    rows = []
    for name in dict.fromkeys(r.proposer for r in results):
        mine = [r for r in results if r.proposer == name]
        scored = [r for r in mine if r.trace]
        totals = [summarize(r.trace) for r in scored]
        proposals = sum(t["proposals"] for t in totals)
        regrets = [r.regret for r in scored if r.regret is not None]
        costs = [r.trace.usage.cost_usd for r in scored if r.trace.usage.cost_usd is not None]
        rows.append({
            "proposer": name,
            "cases": len(mine),
            "errors": len(mine) - len(scored),
            "solved": sum(r.best_g is not None for r in scored),
            "proposals": proposals,
            "verified_rate": sum(t["verified"] for t in totals) / proposals if proposals else None,
            "false_pass_rate": sum(t["false_passes_caught"] for t in totals) / proposals if proposals else None,
            "mean_regret": sum(regrets) / len(regrets) if regrets else None,
            "tool_calls": sum(r.trace.usage.tool_calls for r in scored),
            "tokens": sum(r.trace.usage.input_tokens + r.trace.usage.output_tokens for r in scored),
            "cost_usd": sum(costs) if costs else None,
            "seconds": sum(r.trace.seconds for r in scored),
        })
    return rows
