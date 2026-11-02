"""Design cases for the eval, and an oracle that finds the true optimum of each by brute force."""
from .physics import MATERIALS, STOCK_THICKNESSES_MM, check
from .schemas import Design, Requirement

CASES = {
    "A": Requirement(case="A", load_n=2000, length_mm=150, max_deflection_mm=3.0),
    "B": Requirement(case="B", load_n=5000, length_mm=120, max_deflection_mm=2.0),
    "C": Requirement(case="C", load_n=800, length_mm=250, max_deflection_mm=4.0),
    "D": Requirement(case="D", load_n=3000, length_mm=200, max_deflection_mm=2.5, width_range_mm=(10, 30)),
    # Hard cases, each with a trap. The obvious answer is wrong.
    # E: a higher ultimate factor, so REQ-002 governs, not yield. Sizing for yield alone gives a part that fails.
    "E": Requirement(case="E", load_n=2500, length_mm=140, max_deflection_mm=3.0, ultimate_factor=1.6),
    # F: plate at most 8 mm and a stiff deflection limit. Only steel fits; about 1% of the design space passes.
    "F": Requirement(case="F", load_n=600, length_mm=150, max_deflection_mm=1.5, width_range_mm=(20, 80),
                     thickness_range_mm=(3, 8)),
    # G: thin plate, where 0.5 mm of tolerance is a big share of the thickness. Ignoring it picks a fragile part.
    "G": Requirement(case="G", load_n=600, length_mm=80, max_deflection_mm=4.0, width_range_mm=(20, 80),
                     thickness_range_mm=(3, 8)),
}
STANDARD, HARD = ["A", "B", "C", "D"], ["E", "F", "G"]


def oracle(req: Requirement) -> tuple[Design, float] | None:
    """Every material x every width in 0.1 mm steps x every stock thickness. Returns the lightest design that passes.
    Widths are machined, so any width is allowed; 0.1 mm is finer than a drawing tolerance would care about."""
    (wmin, wmax), (tmin, tmax) = req.width_range_mm, req.thickness_range_mm
    best = None
    for material in MATERIALS:
        for tenths in range(round(wmin * 10), round(wmax * 10) + 1):
            width = tenths / 10
            for thickness in (t for t in STOCK_THICKNESSES_MM if tmin <= t <= tmax):
                d = Design(material=material, width_mm=width, thickness_mm=thickness)
                c = check(d, req)
                if c.passes and (best is None or c.mass_g < best[1]):
                    best = (d, c.mass_g)
    return best
