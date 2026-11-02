"""Ground truth: a solid rectangular cantilever bracket with a load at the tip.

Textbook beam formulas, nothing more. Good enough to check claims, not to fly hardware.
"""
from dataclasses import dataclass

from .schemas import Check, Condition, Design, Requirement, RequirementResult


@dataclass(frozen=True)
class Material:
    density: float            # kg/m^3
    youngs_modulus: float     # Pa
    yield_strength: float     # Pa
    ultimate_strength: float  # Pa


# Typical room-temperature handbook values, not design allowables (real programs use MMPDS A/B-basis data).
MATERIALS = {
    "Al 6061-T6": Material(density=2700, youngs_modulus=68.9e9, yield_strength=276e6, ultimate_strength=310e6),
    "Ti-6Al-4V": Material(density=4430, youngs_modulus=113.8e9, yield_strength=880e6, ultimate_strength=950e6),
    "Steel 4130": Material(density=7850, youngs_modulus=205e9, yield_strength=435e6, ultimate_strength=670e6),
}

# Illustrative metric plate thicknesses. A part machined from plate cannot be thicker than the plate you can buy.
STOCK_THICKNESSES_MM = (3, 4, 5, 6, 8, 10, 12, 16, 20, 25, 30)


def margin(allowable: float, factor: float, applied: float) -> float:
    """Margin of safety as structures teams write it: MS = allowable / (factor x applied) - 1. MS >= 0 passes."""
    return allowable / (factor * applied) - 1


def analyze(design: Design, req: Requirement, load_n: float, thickness_mm: float) -> Condition:
    m = MATERIALS[design.material]
    b, h, length = design.width_mm / 1000, thickness_mm / 1000, req.length_mm / 1000

    stress = 6 * load_n * length / (b * h**2)                               # max bending stress at the root
    deflection = 4 * load_n * length**3 / (m.youngs_modulus * b * h**3)     # tip deflection
    return Condition(
        load_n=load_n,
        thickness_mm=thickness_mm,
        stress_mpa=stress / 1e6,
        margin_yield=margin(m.yield_strength, req.yield_factor, stress),
        margin_ultimate=margin(m.ultimate_strength, req.ultimate_factor, stress),
        deflection_mm=deflection * 1000,
    )


def _holds(c: Condition, req: Requirement) -> bool:
    return c.margin_yield >= 0 and c.margin_ultimate >= 0 and c.deflection_mm <= req.max_deflection_mm


def check(design: Design, req: Requirement) -> Check:
    nominal = analyze(design, req, req.load_n, design.thickness_mm)
    worst = analyze(design, req, req.load_n * (1 + req.load_uncertainty), design.thickness_mm - req.thickness_tolerance_mm)
    m = MATERIALS[design.material]
    mass_g = m.density * design.width_mm * design.thickness_mm * req.length_mm / 1e6
    (wmin, wmax), (tmin, tmax) = req.width_range_mm, req.thickness_range_mm

    results = [
        RequirementResult(id="REQ-001", title="Yield strength", method="analysis",
                          value=f"MS {nominal.margin_yield:+.2f}", limit=f"MS >= 0 at factor {req.yield_factor:g}",
                          passed=nominal.margin_yield >= 0),
        RequirementResult(id="REQ-002", title="Ultimate strength", method="analysis",
                          value=f"MS {nominal.margin_ultimate:+.2f}", limit=f"MS >= 0 at factor {req.ultimate_factor:g}",
                          passed=nominal.margin_ultimate >= 0),
        RequirementResult(id="REQ-003", title="Tip deflection", method="analysis",
                          value=f"{nominal.deflection_mm:.2f} mm", limit=f"<= {req.max_deflection_mm:g} mm",
                          passed=nominal.deflection_mm <= req.max_deflection_mm),
        RequirementResult(id="REQ-004", title="Envelope", method="inspection",
                          value=f"{design.width_mm:g} x {design.thickness_mm:g} mm",
                          limit=f"w {wmin:g}-{wmax:g}, t {tmin:g}-{tmax:g} mm",
                          passed=wmin <= design.width_mm <= wmax and tmin <= design.thickness_mm <= tmax),
        RequirementResult(id="REQ-005", title="Standard plate stock", method="inspection",
                          value=f"t {design.thickness_mm:g} mm", limit="t is a stock plate thickness",
                          passed=design.thickness_mm in STOCK_THICKNESSES_MM),
        RequirementResult(id="REQ-006", title="Robustness", method="analysis",
                          value=f"MS {worst.margin_yield:+.2f}, {worst.deflection_mm:.2f} mm",
                          limit=f"REQ-001..003 hold at +{req.load_uncertainty:.0%} load, -{req.thickness_tolerance_mm:g} mm",
                          passed=_holds(worst, req)),
        RequirementResult(id="REQ-007", title="Proportions", method="inspection",
                          value=f"t/w {design.thickness_mm / design.width_mm:.2f}",
                          limit=f"<= {req.max_depth_to_width:g} (lateral stability proxy)",
                          passed=design.thickness_mm / design.width_mm <= req.max_depth_to_width),
    ]
    return Check(nominal=nominal, worst_case=worst, mass_g=mass_g, results=results,
                 passes=all(r.passed for r in results))
