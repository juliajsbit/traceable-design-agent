"""Compliance Engine: a library of rules, each checked by plain code against one design.

Every rule names the standard it comes from, the value it found, the limit, and how far the
design is from that limit. A model never decides pass or fail here; a separate review agent
only explains the results and proposes actions (review_compliance below).

Illustrative, not a compliance determination: the rules show which questions a regulated
space program asks, which ones this analysis answers, and which ones still need evidence.
"""
from pydantic import BaseModel

from .schemas import Check, Design, Requirement

MATERIAL_SPEC = {
    "Al 6061-T6": "AMS 4027 (6061-T651 plate)",
    "Ti-6Al-4V": "AMS 4911 (annealed plate)",
    "Steel 4130": "AMS 6350 (normalized plate)",
}
# MSFC-STD-3029 rates resistance to stress corrosion cracking; all three of ours are in the high-resistance table
SCC_RESISTANCE = {"Al 6061-T6": "high", "Ti-6Al-4V": "high", "Steel 4130": "high (below 180 ksi strength)"}
# Specialty metals under DFARS 252.225-7009: titanium alloys, and alloy steels with more than 0.25% chromium (4130 has about 1%)
SPECIALTY_METAL = {"Ti-6Al-4V", "Steel 4130"}
FINISH = {
    "Al 6061-T6": ("action", "Specify a finish: chemical conversion coating or anodize.", "MIL-DTL-5541 / MIL-A-8625"),
    "Ti-6Al-4V": ("pass", "Titanium needs no corrosion finish in this environment.", "-"),
    "Steel 4130": ("action", "4130 rusts: specify plating or primer and paint.", "program finish spec"),
}
# Illustrative program minimums for a tested metallic structure; a real program takes them from its standard
MIN_YIELD_FACTOR, MIN_ULTIMATE_FACTOR = 1.25, 1.4


def rule(id_, category, title, standard, status, value="", limit="", margin=None, detail=""):
    """status: pass, fail, action (evidence or work still needed), info. margin: how far from the limit, 0.12 = 12%."""
    return {"id": id_, "category": category, "title": title, "standard": standard, "status": status,
            "value": value, "limit": limit, "margin": margin, "detail": detail}


def evaluate(design: Design, req: Requirement, c: Check) -> list[dict]:
    m, n, w = design.material, c.nominal, c.worst_case
    r = {x.id: x for x in c.results}
    defl_margin = req.max_deflection_mm / n.deflection_mm - 1
    t_over_w = design.thickness_mm / design.width_mm
    blank = (design.width_mm + 10) * (req.length_mm + 10)
    buy_to_fly = blank / (design.width_mm * req.length_mm)
    specialty = m in SPECIALTY_METAL
    return [
        # structural
        rule("S-01", "Structural", "Yield margin of safety", "REQ-001", "pass" if r["REQ-001"].passed else "fail",
             f"MS {n.margin_yield:+.2f}", "MS >= 0", n.margin_yield),
        rule("S-02", "Structural", "Ultimate margin of safety", "REQ-002", "pass" if r["REQ-002"].passed else "fail",
             f"MS {n.margin_ultimate:+.2f}", "MS >= 0", n.margin_ultimate),
        rule("S-03", "Structural", "Margin in the worst case", "REQ-006", "pass" if r["REQ-006"].passed else "fail",
             f"MS {w.margin_yield:+.2f}", "MS >= 0 at +10% load, -0.5 mm", w.margin_yield),
        rule("S-04", "Structural", "Tip deflection", "REQ-003", "pass" if r["REQ-003"].passed else "fail",
             f"{n.deflection_mm:.2f} mm", f"<= {req.max_deflection_mm:g} mm", defl_margin),
        rule("S-05", "Structural", "Factors of safety", "e.g. NASA-STD-5001",
             "pass" if req.yield_factor >= MIN_YIELD_FACTOR and req.ultimate_factor >= MIN_ULTIMATE_FACTOR else "fail",
             f"{req.yield_factor:g} / {req.ultimate_factor:g}", f">= {MIN_YIELD_FACTOR} / {MIN_ULTIMATE_FACTOR}",
             detail="Program minimums here are illustrative; confirm against the program's standard and test approach."),
        rule("S-06", "Structural", "Lateral stability proxy", "REQ-007", "pass" if r["REQ-007"].passed else "fail",
             f"t/w {t_over_w:.2f}", f"<= {req.max_depth_to_width:g}", req.max_depth_to_width / t_over_w - 1,
             "A proportion rule, not a buckling analysis."),
        # materials
        rule("M-01", "Materials", "Design allowables", "MMPDS A- or B-basis", "action", "typical handbook values",
             "statistically based allowables", detail="Replace typical values with allowables for this alloy, temper and thickness."),
        rule("M-02", "Materials", "Stress corrosion cracking", "MSFC-STD-3029", "pass", SCC_RESISTANCE[m] + " resistance",
             "high resistance"),
        rule("M-03", "Materials", "Outgassing in vacuum", "ASTM E595", "pass", "bare metal",
             "TML <= 1.0%, CVCM <= 0.10%", detail="Applies to coatings, adhesives and lubricants added later."),
        rule("M-04", "Materials", "Corrosion protection finish", FINISH[m][2], FINISH[m][0], detail=FINISH[m][1]),
        rule("M-05", "Materials", "Dissimilar metals at the joint", "MIL-STD-889", "action",
             detail="Specify fastener material and finish to avoid galvanic corrosion."),
        rule("M-06", "Materials", "Material specification", MATERIAL_SPEC[m].split(" (")[0], "info", MATERIAL_SPEC[m]),
        # manufacturing
        rule("F-01", "Manufacturing", "Plate you can buy", "REQ-005", "pass" if r["REQ-005"].passed else "fail",
             f"{design.thickness_mm:g} mm", "standard plate"),
        rule("F-02", "Manufacturing", "Buy-to-fly ratio", "cost driver", "pass" if buy_to_fly <= 3 else "action",
             f"{buy_to_fly:.2f}", "<= 3", 3 / buy_to_fly - 1, "Blank with 5 mm stock per side, machined from plate."),
        rule("F-03", "Manufacturing", "Thickness tolerance", "REQ-006", "pass" if r["REQ-006"].passed else "fail",
             f"passes with a {req.thickness_tolerance_mm:g} mm thinner plate", "still passes"),
        # fracture and fatigue
        rule("X-01", "Fracture & fatigue", "Fracture control classification", "e.g. NASA-STD-5019", "action",
             detail="Classify the part, for example non-fracture-critical, and record the rationale."),
        rule("X-02", "Fracture & fatigue", "Fatigue", "program loads", "action",
             detail="Not analyzed here. Needs cyclic load cases such as launch vibration."),
        # procurement and regulatory
        rule("P-01", "Procurement", "Specialty metals (DoD work)", "DFARS 252.225-7009", "action" if specialty else "pass",
             "specialty metal" if specialty else "not a specialty metal",
             detail="Melt must come from the US or a qualifying country; the Supplier Desk asks every vendor."
             if specialty else "Aluminum alloys are not restricted."),
        rule("P-02", "Procurement", "Material traceability", "purchase order clause", "action",
             detail="Certificate of conformance and mill test report traceable to the heat lot."),
        rule("P-03", "Procurement", "Supplier quality system", "AS9100", "action",
             detail="The Supplier Desk only accepts AS9100 vendors."),
        rule("P-04", "Procurement", "Export control", "ITAR / EAR", "info", detail="Depends on the end use; not determined here."),
    ]


def supplier_requirements(design: Design) -> dict:
    """What the Compliance Engine hands to the Supplier Desk."""
    return {"as9100": True, "dfars_melt": design.material in SPECIALTY_METAL}


def summary(items: list[dict]) -> dict:
    count = {s: sum(i["status"] == s for i in items) for s in ("pass", "fail", "action", "info")}
    checked = count["pass"] + count["fail"] + count["action"]
    return {**count, "score": count["pass"] / checked if checked else None}


# ---- review agent: explains the results, never changes them ----

class Action(BaseModel):
    rule_id: str
    owner: str   # e.g. stress, materials, manufacturing, procurement
    action: str


class ComplianceReview(BaseModel):
    summary: str
    top_risks: list[str]
    actions: list[Action]


def review_prompt(design: Design, items: list[dict]) -> str:
    rows = "\n".join(f"{i['id']} [{i['status']}] {i['title']} ({i['standard']}): {i['value']} {i['detail']}".strip() for i in items)
    return (f"You are the compliance engineer for a space bracket: {design.material}, {design.width_mm:g} x "
            f"{design.thickness_mm:g} mm. Below are rule results from the compliance engine. Use only these results, "
            "cite rule ids, and do not change any status.\n\n" + rows +
            "\n\nWrite a two-sentence summary, the top three risks, and up to five concrete actions with an owner.")
