"""Requirements Desk: a mission brief in plain words becomes a Requirement a person approves.

The agent reads the brief and fills a draft: every value comes with its source, a quote from
the brief or a stated assumption. Code does the arithmetic (load = mass x g x 9.81), fills the
defaults it has to assume, and checks the result makes sense. The graph then stops with
LangGraph interrupt() until a person approves or edits it (team.py, node "approve").
"""
from typing import Literal

from pydantic import BaseModel

from .schemas import Requirement

G = 9.81
SAMPLE_BRIEF = (
    "We need a bracket that holds a 4 kg star tracker on a small satellite. It mounts to the side panel and the "
    "tracker sits 150 mm out from the panel. Launch is a rideshare; use 15 g quasi-static for design. The tracker "
    "needs a stiff mount: tip deflection under load must stay below 1.5 mm. It sits under a solar array hinge, "
    "so the plate can be at most 8 mm thick, and there is room for about 80 mm of width. The program uses "
    "factors of 1.25 on yield and 1.4 on ultimate."
)


class FieldSource(BaseModel):
    field: str
    value: str
    source: Literal["brief", "assumption"]
    note: str            # the quote from the brief, or why this was assumed


class RequirementDraft(BaseModel):
    part: str
    payload_mass_kg: float | None = None
    design_load_g: float | None = None
    load_n: float | None = None          # only if the brief gives a force directly
    length_mm: float | None = None
    max_deflection_mm: float | None = None
    width_max_mm: float | None = None
    thickness_max_mm: float | None = None
    yield_factor: float | None = None
    ultimate_factor: float | None = None
    sources: list[FieldSource]
    open_questions: list[str]


def prompt(brief: str) -> str:
    return (
        "You are a structures engineer turning a mission brief into design requirements for a cantilever bracket "
        "(fixed at one end, load at the tip, solid rectangular section machined from plate).\n\n"
        f"Brief:\n{brief}\n\n"
        "Fill only what the brief supports. If the brief gives a mass and an acceleration, give both and do not "
        "compute the force yourself. For every field you fill, add a source: 'brief' with the exact quote, or "
        "'assumption' with the reason. Leave a field empty rather than guess. List open questions an engineer "
        "should ask before signing off."
    )


def build(draft: RequirementDraft, case: str = "brief") -> tuple[Requirement, list[str], list[str], list[str]]:
    """Turn the draft into a Requirement. Returns (requirement, derived, assumed, problems)."""
    defaults = Requirement()
    derived, assumed, problems = [], [], []
    if draft.load_n:
        load = draft.load_n
    elif draft.payload_mass_kg and draft.design_load_g:
        load = draft.payload_mass_kg * draft.design_load_g * G
        derived.append(f"load = {draft.payload_mass_kg:g} kg x {draft.design_load_g:g} g x {G} = {load:.0f} N")
    else:
        load = defaults.load_n
        assumed.append(f"load {load:g} N (no mass and acceleration in the brief)")

    def pick(name, value, default):
        if value is None:
            assumed.append(f"{name} {default:g} (default)")
            return default
        return value

    length = pick("length_mm", draft.length_mm, defaults.length_mm)
    deflection = pick("max_deflection_mm", draft.max_deflection_mm, defaults.max_deflection_mm)
    width_max = pick("width_max_mm", draft.width_max_mm, defaults.width_range_mm[1])
    thickness_max = pick("thickness_max_mm", draft.thickness_max_mm, defaults.thickness_range_mm[1])
    yf = pick("yield_factor", draft.yield_factor, defaults.yield_factor)
    uf = pick("ultimate_factor", draft.ultimate_factor, defaults.ultimate_factor)

    width_min, thickness_min = defaults.width_range_mm[0], defaults.thickness_range_mm[0]
    if width_max <= width_min:
        problems.append(f"width limit {width_max:g} mm is below the {width_min:g} mm minimum")
    if thickness_max <= thickness_min:
        problems.append(f"thickness limit {thickness_max:g} mm is below the {thickness_min:g} mm minimum")
    if not 0 < deflection < length:
        problems.append(f"deflection limit {deflection:g} mm does not make sense for a {length:g} mm arm")
    if uf < yf:
        problems.append("ultimate factor is lower than the yield factor")

    req = Requirement(case=case, part=draft.part or defaults.part, load_n=round(load, 1), length_mm=length,
                      max_deflection_mm=deflection, width_range_mm=(width_min, width_max),
                      thickness_range_mm=(thickness_min, thickness_max), yield_factor=yf, ultimate_factor=uf)
    return req, derived, assumed, problems
