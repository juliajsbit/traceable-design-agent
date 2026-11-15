"""The same agent as AgentProposer, built with LangGraph, so the two can be compared.

    START -> model -> tools -> model -> ... -> finalize -> END

model:    asks the model what to do next
tools:    runs check_design when the model asked for it
finalize: turns the conversation into structured proposals

On a team (team.py) the same agent works as a specialist for one material and reports
every step through on_event, which is what the live dashboard draws.
"""
import json
from types import SimpleNamespace

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .physics import check
from .proposers import CHECK_TOOL, _ModelProposer, build_prompt, run_check_tool
from .schemas import Design, ProposalBatch, ProposalDraft, Requirement
from .verify import label


def check_event(design: Design, req: Requirement) -> dict:
    """What the dashboard needs to know about one physics check."""
    c = check(design, req)
    return {"design": design.model_dump(), "label": label(c), "passes": c.passes, "mass_g": c.mass_g,
            "margin_nominal": c.nominal.margin_yield, "margin_worst": c.worst_case.margin_yield,
            "deflection_mm": c.nominal.deflection_mm, "failed": [r.id for r in c.results if not r.passed]}


class LangGraphProposer(_ModelProposer):
    name = "langgraph"

    def __init__(self, *args, max_turns: int = 20, chat_model=None, focus: str | None = None,
                 note: str | None = None, on_event=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns
        self.chat_model = chat_model
        self.focus = focus        # the one material this agent works on, when it is on a team
        self.note = note          # guidance from the lead agent
        self.on_event = on_event or (lambda event: None)

    def _emit(self, type_: str, **data) -> None:
        self.on_event({"type": type_, "agent": self.focus or self.name, **data})

    def _count(self, message) -> None:
        """Feed a LangChain message into the same usage and refusal checks as the other proposers."""
        u = message.usage_metadata or {}
        meta = message.response_metadata or {}
        response = SimpleNamespace(model=meta.get("model_name"), stop_reason=meta.get("stop_reason"), stop_details=None,
                                   usage=SimpleNamespace(input_tokens=u.get("input_tokens", 0),
                                                         output_tokens=u.get("output_tokens", 0)))
        self._track(response)
        self._refused(response)
        self._emit("usage", **self.usage.model_dump(exclude={"served_by"}))

    def _build(self, req: Requirement):
        llm = self.chat_model or ChatAnthropic(model=self.model, max_tokens=16000)

        @tool("check_design", args_schema=Design, description=CHECK_TOOL["description"])
        def check_design(material: str, width_mm: float, thickness_mm: float) -> str:
            self.usage.tool_calls += 1
            args = {"material": material, "width_mm": width_mm, "thickness_mm": thickness_mm}
            result = run_check_tool(args, req)
            if "error" not in json.loads(result):
                self._emit("check", **check_event(Design(**args), req))
            return result

        llm_with_tools = llm.bind_tools([check_design])
        structured = llm.with_structured_output(ProposalBatch, include_raw=True)

        def model(state: MessagesState) -> dict:
            response = llm_with_tools.invoke(state["messages"])
            self._count(response)
            text = response.content if isinstance(response.content, str) else " ".join(
                b.get("text", "") for b in response.content if isinstance(b, dict) and b.get("type") == "text")
            self._emit("turn", tool_calls=len(response.tool_calls), text=text.strip()[:400])
            return {"messages": [response]}

        def finalize(state: MessagesState) -> dict:
            ask = HumanMessage("Return your final proposals now.")
            result = structured.invoke(state["messages"] + [ask])
            self._count(result["raw"])
            if result["parsing_error"]:  # e.g. empty numbers when nothing passed; ask once more, plainly
                ask = HumanMessage("Return your final proposals now. Every proposal needs a number for each claim, "
                                   "even for designs that fail: give your best estimate.")
                result = structured.invoke(state["messages"] + [ask])
                self._count(result["raw"])
                if result["parsing_error"]:
                    raise result["parsing_error"]
            self.proposals = result["parsed"].proposals
            return {"messages": [ask, result["raw"]]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", model)
        graph.add_node("tools", ToolNode([check_design]))
        graph.add_node("finalize", finalize)
        graph.add_edge(START, "model")
        # tools_condition routes to "tools" when the model asked for one, otherwise to END, which we map to finalize
        graph.add_conditional_edges("model", tools_condition, {"tools": "tools", END: "finalize"})
        graph.add_edge("tools", "model")
        graph.add_edge("finalize", END)
        return graph.compile()

    def propose(self, req: Requirement, feedback: list[str]) -> list[ProposalDraft]:
        prompt = build_prompt(req, feedback, self.n, with_tool=True)
        if self.focus:
            prompt += (f"\nYou are the {self.focus} specialist on a team. Propose only {self.focus} designs; "
                       "other agents cover the other materials. Search step by step: check 2 or 3 candidates at a "
                       "time, read the results, then narrow toward the lightest design that passes. Check about "
                       "10 designs in total before you answer.")
        if self.note:
            prompt += f"\nNote from the lead engineer: {self.note}"
        # each model turn is two graph steps (model, tools), plus the first and last steps
        self._build(req).invoke({"messages": [HumanMessage(prompt)]}, {"recursion_limit": 2 * self.max_turns + 2})
        return self.proposals
