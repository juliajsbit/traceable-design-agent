from typing import Literal

from pydantic import BaseModel

MaterialName = Literal["Al 6061-T6", "Ti-6Al-4V", "Steel 4130"]
Label = Literal["verified", "fails", "fragile", "not manufacturable", "out of envelope"]


class Requirement(BaseModel):
    """One design case. Factors of safety are illustrative; a real program takes them from its standard."""
    case: str = "A"
    part: str = "cantilever bracket, load at the tip"
    load_n: float = 2000
    length_mm: float = 150
    yield_factor: float = 1.25
    ultimate_factor: float = 1.4
    max_deflection_mm: float = 3.0
    width_range_mm: tuple[float, float] = (10, 60)
    thickness_range_mm: tuple[float, float] = (3, 30)
    load_uncertainty: float = 0.10       # robustness check: load can be 10% higher
    thickness_tolerance_mm: float = 0.5  # robustness check: plate can be this much thinner
    max_depth_to_width: float = 2.0      # proxy for lateral stability; no buckling analysis is done


class Design(BaseModel):
    material: MaterialName
    width_mm: float
    thickness_mm: float


class ProposalDraft(BaseModel):
    """What a proposer returns: a design plus what it claims about that design."""
    design: Design
    claimed_margin_yield: float
    claimed_deflection_mm: float
    claimed_mass_g: float
    rationale: str


class ProposalBatch(BaseModel):
    proposals: list[ProposalDraft]


class Proposal(ProposalDraft):
    id: str
    round: int


class Condition(BaseModel):
    """Beam analysis at one load and thickness."""
    load_n: float
    thickness_mm: float
    stress_mpa: float
    margin_yield: float
    margin_ultimate: float
    deflection_mm: float


class RequirementResult(BaseModel):
    """One row of the verification matrix."""
    id: str
    title: str
    method: Literal["analysis", "inspection"]
    value: str
    limit: str
    passed: bool


class Check(BaseModel):
    """What the physics says. This is the evidence, not the proposer's opinion."""
    nominal: Condition
    worst_case: Condition
    mass_g: float
    results: list[RequirementResult]
    passes: bool


class Finding(BaseModel):
    kind: Literal["false_pass", "claim_off", "out_of_bounds"]
    message: str


class Record(BaseModel):
    proposal: Proposal
    check: Check | None
    findings: list[Finding]
    status: Literal["accepted", "rejected"]
    label: Label
    reason: str


class Usage(BaseModel):
    llm_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None
    served_by: list[str] = []  # models that actually answered; a server-side fallback can differ from the one asked for


class DecisionTrace(BaseModel):
    requirement: Requirement
    proposer: str
    model: str | None = None
    rounds: int
    records: list[Record]
    selected_id: str | None
    selection_reason: str
    usage: Usage = Usage()
    seconds: float = 0
    created_at: str
