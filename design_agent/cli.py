import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from .cases import CASES, STANDARD
from .evals import CaseResult, rescore, run_eval, scoreboard
from .pipeline import run, summarize
from .proposers import DEFAULT_MODEL, PROPOSERS
from .report import render, render_eval


def make(name: str, model: str):
    return PROPOSERS[name]() if name == "naive" else PROPOSERS[name](model=model)


def needs_key(names) -> bool:
    if any(n != "naive" for n in names) and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set. Put it in .env (see .env.example). The naive proposer needs no key.", file=sys.stderr)
        return True
    return False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Propose bracket designs, verify every claim, keep the evidence.")
    sub = parser.add_subparsers(dest="command", required=True)

    one = sub.add_parser("run", help="one proposer on one case, with an HTML decision trace")
    one.add_argument("--proposer", choices=list(PROPOSERS), default="naive")
    one.add_argument("--case", choices=list(CASES), default="A")
    one.add_argument("--rounds", type=int, default=1, help="propose, verify, feed back; repeat")

    ev = sub.add_parser("eval", help="every proposer on every case, with a scoreboard")
    ev.add_argument("--proposers", nargs="+", choices=list(PROPOSERS), default=list(PROPOSERS))
    ev.add_argument("--cases", nargs="+", choices=list(CASES), default=STANDARD, help="A-D standard, E-G hard")
    ev.add_argument("--concurrency", type=int, default=4)

    rs = sub.add_parser("rescore", help="score saved eval results again and rebuild the report, no model calls")
    rs.add_argument("--out", type=Path, default=Path("results") / "eval")

    for p in (one, ev):
        p.add_argument("--model", default=os.environ.get("DESIGN_MODEL") or DEFAULT_MODEL)
        p.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    load_dotenv()

    if args.command == "run":
        if needs_key([args.proposer]):
            return 1
        trace = run(CASES[args.case], make(args.proposer, args.model), rounds=args.rounds)
        out = args.out or Path("results") / f"{args.proposer}_{args.case}"
        out.mkdir(parents=True, exist_ok=True)
        (out / "trace.json").write_text(trace.model_dump_json(indent=2) + "\n")
        (out / "report.html").write_text(render(trace))
        print(json.dumps(summarize(trace)))
        print(f"{trace.selected_id}: {trace.selection_reason}")
        print(f"report: {out / 'report.html'}")
        return 0

    if args.command == "rescore":
        out = args.out
        results = rescore([CaseResult.model_validate_json(line) for line in (out / "results.jsonl").read_text().splitlines()])
    else:
        if needs_key(args.proposers):
            return 1
        proposers = {name: partial(make, name, args.model) for name in args.proposers}
        results = run_eval({c: CASES[c] for c in args.cases}, proposers, concurrency=args.concurrency)
        out = args.out or Path("results") / "eval"
        out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text("".join(r.model_dump_json() + "\n" for r in results))
    (out / "report.html").write_text(render_eval(results))
    for row in scoreboard(results):
        print(json.dumps(row))
    print(f"report: {out / 'report.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
