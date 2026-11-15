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

    tm = sub.add_parser("team", help="lead agent plus one specialist per material in parallel; records the run for replay")
    tm.add_argument("--case", choices=list(CASES), default="E")
    tm.add_argument("--n", type=int, default=4, help="proposals per specialist")
    tm.add_argument("--brief", help="start from a mission brief instead of a case; 'sample' uses the built-in one")
    tm.add_argument("--base", help="a change run: start from this recorded run and run only what the change touches")
    tm.add_argument("--set", nargs="*", default=[], metavar="FIELD=VALUE",
                    help="with --base: what changed, for example load_n=2400 length_mm=160 qty=50")

    pk = sub.add_parser("package", help="build the Design Package of a recorded run: one HTML document, no model calls")
    pk.add_argument("run", help="a recording name from results/live, without .jsonl")
    pk.add_argument("--out", type=Path)

    sv = sub.add_parser("serve", help="live dashboard at http://127.0.0.1:8765")
    sv.add_argument("--port", type=int, default=8765)

    rs = sub.add_parser("rescore", help="score saved eval results again and rebuild the report, no model calls")
    rs.add_argument("--out", type=Path, default=Path("results") / "eval")

    for p in (one, ev, tm):
        p.add_argument("--model", default=os.environ.get("DESIGN_MODEL") or DEFAULT_MODEL)
        p.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    load_dotenv()

    if args.command == "serve":
        import uvicorn
        uvicorn.run("design_agent.server:app", host="127.0.0.1", port=args.port)
        return 0

    if args.command == "package":
        from . import ledger, package
        source = Path("results") / "live" / f"{args.run}.jsonl"
        if not source.exists():
            print(f"no recording {source}")
            return 1
        events = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        out = args.out or Path("results") / "packages" / f"design-package-{args.run}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(package.build(events, args.run))
        chain = ledger.verify(events)
        print(f"ledger: {chain['chained']} of {chain['events']} events chained, " +
              (f"intact, head {chain['head']}" if chain["ok"] else f"BROKEN at event {chain['broken_at']}" if chain["chained"] else "no chain"))
        print(f"package: {out}")
        return 0

    if args.command == "team":
        if needs_key(["team"]):
            return 1
        from datetime import datetime
        from .team import run_team
        name = "change" if args.base else "brief" if args.brief else args.case
        out = args.out or Path("results") / "live" / f"{datetime.now():%Y%m%d-%H%M%S}_{name}_{args.model.removeprefix('claude-')}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            def emit(event):
                f.write(json.dumps(event) + "\n")
                f.flush()
                if event["type"] in ("requirements", "plan", "agent_done", "agent_error", "decision", "review", "run_end", "impact", "step"):
                    print(json.dumps({k: v for k, v in event.items() if k not in ("matrix", "requirement", "findings", "pool")})[:300])
            from .requirements_desk import SAMPLE_BRIEF
            brief = SAMPLE_BRIEF if args.brief == "sample" else args.brief
            from . import impact as im
            if args.base:
                from .schemas import Requirement
                baseline = im.load(out.parent, args.base)
                if baseline is None:
                    print(f"no snapshot for {args.base} in {out.parent}")
                    return 1
                edits = dict(pair.split("=", 1) for pair in args.set)
                snap = run_team(im.apply_edits(Requirement(**baseline["req"]), edits), args.model, emit, n=args.n,
                                baseline=baseline, qty=int(edits.get("qty", baseline["qty"])))
            else:
                snap = run_team(None if brief else CASES[args.case], args.model, emit, n=args.n, brief=brief)  # approves as drafted
            im.save(out.parent, out.stem, snap, args.base)
        print(f"recording: {out}")
        return 0

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
