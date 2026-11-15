"""Live dashboard: runs the design team and streams every event to the browser.

    /                 the dashboard page
    /api/info         cases, models and saved recordings
    /api/run          run the team live (needs ANTHROPIC_API_KEY), streams events and records them;
                      with brief=..., the Requirements Desk runs first and the run waits for /api/approve;
                      with base=<run> and edits=<json>, a change run: it starts from that run's snapshot
                      and only the steps that read the change run again
    /api/approve      approve (or edit) the drafted requirement of a waiting run
    /api/replay       stream a saved recording at its original pace, no API calls
    /api/package      the Design Package of a saved recording: one HTML document, built with no API calls
"""
import json
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from . import impact as im
from . import package
from .cases import CASES
from .proposers import PRICE_PER_MTOK
from .requirements_desk import SAMPLE_BRIEF
from .schemas import Requirement
from .team import run_team

STATIC = Path(__file__).parent / "static"
RECORDINGS = Path("results") / "live"

load_dotenv()
app = FastAPI(title="Design team")
WAITING: dict[str, dict] = {}   # run id -> {"event": threading.Event, "decision": dict}
APPROVAL_TIMEOUT_S = 15 * 60
REPLAY_MAX_GAP_S = 2.5            # a long wait for a person is shortened in replays


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event)}\n\n"


@app.get("/")
def page():
    return FileResponse(STATIC / "dashboard.html", headers={"Cache-Control": "no-store"})  # always serve the latest page


@app.get("/api/info")
def info():
    recordings = sorted(RECORDINGS.glob("*.jsonl"), reverse=True) if RECORDINGS.exists() else []
    return {"cases": {k: v.model_dump() for k, v in CASES.items()}, "models": list(PRICE_PER_MTOK),
            "recordings": [p.stem for p in recordings], "sample_brief": SAMPLE_BRIEF,
            "snapshots": [p.name.removesuffix(".snapshot.json") for p in RECORDINGS.glob("*.snapshot.json")]}


@app.get("/api/run")
def run(case: str = "E", model: str = "claude-haiku-5-5", n: int = 4, brief: str | None = None,
        base: str | None = None, edits: str | None = None):
    if (not brief and not base and case not in CASES) or model not in PRICE_PER_MTOK:
        raise HTTPException(400, "unknown case or model")
    baseline, req, qty = None, None if brief else CASES.get(case), 10
    if base:
        baseline, brief = im.load(RECORDINGS, base), None
        if baseline is None:
            raise HTTPException(404, "no snapshot for that run")
        try:
            changed = json.loads(edits or "{}")
            req, qty = im.apply_edits(Requirement(**baseline["req"]), changed), int(changed.get("qty", baseline["qty"]))
            if qty < 1:
                raise ValueError("quantity must be at least 1")
        except (ValueError, TypeError, AttributeError) as e:
            raise HTTPException(400, f"bad edits: {e}")
    events: queue.Queue = queue.Queue()
    RECORDINGS.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}_{'change' if base else 'brief' if brief else case}_{model.removeprefix('claude-')}"
    path = RECORDINGS / f"{run_id}.jsonl"

    def wait_for_person(payload: dict) -> dict:
        slot = WAITING[run_id] = {"event": threading.Event(), "decision": {"edits": {}}}
        emit_live({"type": "waiting", "run_id": run_id})
        slot["event"].wait(APPROVAL_TIMEOUT_S)
        return WAITING.pop(run_id, slot)["decision"]

    def emit_live(event):   # sent to the browser only, not recorded
        events.put(event)

    def work():
        with path.open("w") as f:
            def emit(event):
                f.write(json.dumps(event) + "\n")
                f.flush()
                events.put(event)
            try:
                snap = run_team(req, model, emit, n=n, brief=brief, approve=wait_for_person, baseline=baseline, qty=qty)
                im.save(RECORDINGS, run_id, snap, base)
            except Exception as e:
                emit({"type": "error", "error": f"{type(e).__name__}: {e}"})
        events.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (event := events.get()) is not None:
            yield _sse(event)
        yield _sse({"type": "recorded", "name": path.stem})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/api/approve/{run_id}")
def approve(run_id: str, decision: dict = Body(default={})):
    slot = WAITING.get(run_id)
    if not slot:
        raise HTTPException(404, "no run is waiting for approval")
    slot["decision"] = {"edits": decision.get("edits") or {}}
    slot["event"].set()
    return {"ok": True}


def _recording(name: str) -> list[dict]:
    path = (RECORDINGS / f"{name}.jsonl").resolve()
    if path.parent != RECORDINGS.resolve() or not path.exists():
        raise HTTPException(404, "no such recording")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@app.get("/api/package")
def design_package(name: str, download: bool = False):
    headers = {"Content-Disposition": f'attachment; filename="design-package-{name}.html"'} if download else {}
    return HTMLResponse(package.build(_recording(name), name), headers=headers)


@app.get("/api/replay")
def replay(name: str, speed: float = 1.0):
    lines = _recording(name)

    def stream():
        previous = 0.0
        for event in lines:
            t = event.get("t", previous)
            gap = min(max(t - previous, 0), REPLAY_MAX_GAP_S) / max(speed, 0.1)
            if gap > 0:
                time.sleep(gap)
            previous = t
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream")
