# Working rules for coding agents in this repo

## Setup and checks
- Python 3.12, project venv: `python3.12 -m venv venv && ./venv/bin/pip install -r requirements.txt`
- Tests (offline, no API keys): `./venv/bin/python -m pytest -q`
- Lint: `./venv/bin/pylint --errors-only $(git ls-files '*.py')`
- A change is done only when both pass. CI runs the same two commands.

## Ground truth
- `design_agent/physics.py` and `design_agent/verify.py` decide whether a design passes. Model output never does.
- Do not change a formula, a material value or a factor of safety without a test that pins the new number
  against a hand calculation.
- `design_agent/review.py` `decide()` has the last word over the Chief Engineer agent. Keep it that way.
- New agent output must be a Pydantic model and must be checked by code before anything acts on it.

## Secrets and cost
- Never open, print or commit `.env`. Keys stay there.
- Tests must not call a model. Use the fakes in `tests/`.
- Runs that call a paid model (`cli team`, `cli eval`, Run live) need the owner's approval.

## Data
- Recordings in `results/live/` are evidence: do not edit them. The ledger hash chain will flag it.
- Vendor data in `design_agent/data/` is sample data, not real suppliers.

## Style
- Plain words, short functions, comments that say why.
- Node names in `team.py` are the public names (Studio, LangSmith). Event names feed the dashboard and
  old recordings, so do not rename them.
