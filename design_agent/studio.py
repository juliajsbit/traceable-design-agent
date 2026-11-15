"""The team graph for LangGraph Studio. Run `langgraph dev` in the project folder, then give it {"case": "F"},
or {"brief": "..."} to start with the Requirements Desk: the graph then stops for your approval (interrupt)."""
import os

from design_agent.team import build_team  # Studio loads this file by path, not as part of the package

team, _ = build_team(os.environ.get("DESIGN_MODEL") or "claude-haiku-5-5", n=4, emit=lambda event: None, approval=True)
