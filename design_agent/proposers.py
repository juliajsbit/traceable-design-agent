"""Proposers suggest designs and make claims about them. The verifier decides what is true.

The competitors:
- naive:     fast grid search with a classic mistake
- llm:       a model proposes designs from the requirement alone
- agent:     the same model, but it can call the physics check as a tool before it commits
- langgraph: the same agent built with LangGraph instead of a plain loop (graph_agent.py)
"""
import json
from itertools import product

from .physics import MATERIALS, STOCK_THICKNESSES_MM, analyze, check
from .schemas import Design, ProposalBatch, ProposalDraft, Requirement, Usage

DEFAULT_MODEL = "claude-opus-5-5"
PRICE_PER_MTOK = {  # input, output in USD
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}


class NaiveProposer:
    """Sizes the part for pure tension (stress = F / area) and forgets the bending moment.
    Stands in for any generator that is quick, confident and sometimes wrong."""

    name = "naive"
    model = None

    def __init__(self):
        self.usage = Usage()

    def propose(self, req: Requirement, feedback: list[str]) -> list[ProposalDraft]:
        drafts = []
        for material, width, thickness in product(MATERIALS, (20, 35), (10, 16, 22)):
            d = Design(material=material, width_mm=width, thickness_mm=thickness)
            m = MATERIALS[material]
            naive_stress = req.load_n / (width * thickness / 1e6)  # the bug: no bending
            nominal = analyze(d, req, req.load_n, thickness)
            drafts.append(ProposalDraft(
                design=d,
                claimed_margin_yield=m.yield_strength / (req.yield_factor * naive_stress) - 1,
                claimed_deflection_mm=nominal.deflection_mm,
                claimed_mass_g=check(d, req).mass_g,
                rationale="grid search sized for tensile stress",
            ))
        return drafts


def requirement_text(req: Requirement) -> str:
    (wmin, wmax), (tmin, tmax) = req.width_range_mm, req.thickness_range_mm
    materials = {name: {"density_kg_m3": m.density, "youngs_modulus_gpa": m.youngs_modulus / 1e9,
                        "yield_mpa": m.yield_strength / 1e6, "ultimate_mpa": m.ultimate_strength / 1e6}
                 for name, m in MATERIALS.items()}
    return "\n".join([
        f"Part: {req.part}. Solid rectangular cross-section, width across the load, thickness in the load direction.",
        f"Load {req.load_n:g} N at the tip, length {req.length_mm:g} mm.",
        f"REQ-001 yield margin of safety >= 0 with factor {req.yield_factor:g} (MS = allowable / (factor x stress) - 1).",
        f"REQ-002 ultimate margin of safety >= 0 with factor {req.ultimate_factor:g}.",
        f"REQ-003 tip deflection <= {req.max_deflection_mm:g} mm.",
        f"REQ-004 width {wmin:g}-{wmax:g} mm, thickness {tmin:g}-{tmax:g} mm.",
        f"REQ-005 thickness must be a stock plate thickness: {list(STOCK_THICKNESSES_MM)} mm.",
        f"REQ-006 REQ-001 to REQ-003 must still hold with the load {req.load_uncertainty:.0%} higher "
        f"and the plate {req.thickness_tolerance_mm:g} mm thinner.",
        f"REQ-007 thickness / width <= {req.max_depth_to_width:g} (a proxy for lateral stability).",
        f"Materials: {json.dumps(materials)}",
        "Goal: the lowest mass that meets every requirement.",
    ])


def build_prompt(req: Requirement, feedback: list[str], n: int, with_tool: bool = False) -> str:
    parts = [
        "You are helping size a structural bracket.",
        requirement_text(req),
        f"Propose {n} different designs. For each one, report your own estimate of its yield margin of safety, "
        "tip deflection in mm and mass in grams. Every estimate will be checked independently.",
    ]
    if with_tool:
        parts.append("You can call check_design to analyze a design before you propose it. "
                     "When you are done, answer with the final proposals.")
    if feedback:
        parts.append("Results of your previous round:\n" + "\n".join(feedback))
        parts.append("Use them: avoid repeating rejected designs and look for lighter designs that pass.")
    return "\n\n".join(parts)


class _ModelProposer:
    def __init__(self, model: str = DEFAULT_MODEL, n: int = 8, client=None):
        self.model = model
        self.n = n
        self.usage = Usage(cost_usd=0.0 if model in PRICE_PER_MTOK else None)
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _track(self, response) -> None:
        u = response.usage
        self.usage.llm_calls += 1
        self.usage.input_tokens += u.input_tokens
        self.usage.output_tokens += u.output_tokens
        served = getattr(response, "model", None) or self.model
        if served not in self.usage.served_by:
            self.usage.served_by.append(served)
        if self.model in PRICE_PER_MTOK:
            price_in, price_out = PRICE_PER_MTOK[self.model]
            self.usage.cost_usd += (u.input_tokens * price_in + u.output_tokens * price_out) / 1e6

    @staticmethod
    def _refused(response) -> None:
        if response.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {response.stop_details}")


class LLMProposer(_ModelProposer):
    name = "llm"

    def propose(self, req: Requirement, feedback: list[str]) -> list[ProposalDraft]:
        response = self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            messages=[{"role": "user", "content": build_prompt(req, feedback, self.n)}],
            output_format=ProposalBatch,
            **FALLBACK,
        )
        self._track(response)
        self._refused(response)
        return response.parsed_output.proposals


CHECK_TOOL = {
    "name": "check_design",
    "description": "Run the beam analysis for one design. Returns stress, margins of safety, deflection, mass, "
                   "the worst case for REQ-006, and pass or fail for every requirement.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "material": {"type": "string", "enum": list(MATERIALS)},
            "width_mm": {"type": "number"},
            "thickness_mm": {"type": "number"},
        },
        "required": ["material", "width_mm", "thickness_mm"],
        "additionalProperties": False,
    },
}


def run_check_tool(args: dict, req: Requirement) -> str:
    try:
        c = check(Design(**args), req)
    except Exception as e:  # a bad call is returned to the agent as data, not raised
        return json.dumps({"error": str(e)})
    return c.model_dump_json()


class AgentProposer(_ModelProposer):
    """A manual agent loop: the model may call check_design as many times as it wants,
    then ends with structured proposals. Same loop as lesson 1, with a real model."""

    name = "agent"

    def __init__(self, *args, max_turns: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_turns = max_turns

    def propose(self, req: Requirement, feedback: list[str]) -> list[ProposalDraft]:
        import anthropic

        messages = [{"role": "user", "content": build_prompt(req, feedback, self.n, with_tool=True)}]
        schema = anthropic.transform_schema(ProposalBatch)
        for _ in range(self.max_turns):
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                tools=[CHECK_TOOL],
                messages=messages,
                output_config={"format": {"type": "json_schema", "schema": schema}},
                **FALLBACK,
            )
            self._track(response)
            self._refused(response)
            if response.stop_reason != "tool_use":
                text = next(b.text for b in response.content if b.type == "text")
                return ProposalBatch.model_validate_json(text).proposals

            messages.append({"role": "assistant", "content": response.content})
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    self.usage.tool_calls += 1
                    results.append({"type": "tool_result", "tool_use_id": block.id,
                                    "content": run_check_tool(block.input, req)})
            messages.append({"role": "user", "content": results})
        raise RuntimeError(f"agent did not finish in {self.max_turns} turns")


def _langgraph(*args, **kwargs):
    from .graph_agent import LangGraphProposer  # imported late: it imports from this module
    return LangGraphProposer(*args, **kwargs)


PROPOSERS = {"naive": NaiveProposer, "llm": LLMProposer, "agent": AgentProposer, "langgraph": _langgraph}
