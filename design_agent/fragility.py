"""Fragility and trade-offs: plain code, no model.

Monte Carlo: the analysis uses one load, one plate thickness and one strength. Real parts scatter.
Each sample draws a load, a plate thickness, a width, a yield and ultimate strength and a modulus,
then checks strength and stiffness. The share of samples that fail is the probability of failure,
and the correlation of each input with the governing margin says which scatter matters most.

Trade-offs: every verified design gets a mass, a cost estimate, a worst-case margin and a
probability of failure, so a person can weigh them instead of taking "lightest wins".
"""
import math
import random

from .physics import MATERIALS, check
from .schemas import Design, Requirement

SAMPLES = 2000
# One standard deviation of each input, illustrative
SCATTER = {"load": 0.05, "thickness_mm": 0.15, "width_mm": 0.05, "strength": 0.04, "modulus": 0.02}
INPUTS = list(SCATTER)
# A neutral shop for comparing designs by cost (the Supplier Desk gets real quotes for the chosen one)
REFERENCE_SHOP = {"certifications": ["AS9100D"], "rate_usd_h": 110, "margin": 0.15, "base_lead_days": 8, "shop_load": 0.5, "stock": {}}


def governing_margin(load, b, h, L, yield_s, ult_s, E, req: Requirement) -> float:
    """The smallest of the yield margin, the ultimate margin and the stiffness margin. Below 0 the part fails."""
    stress = 6 * load * L / (b * h * h)
    deflection = 4 * load * L ** 3 / (E * b * h ** 3) * 1000
    return min(yield_s / (req.yield_factor * stress) - 1, ult_s / (req.ultimate_factor * stress) - 1,
               req.max_deflection_mm / deflection - 1)


def monte_carlo(design: Design, req: Requirement, samples: int = SAMPLES, seed: int = 7) -> dict:
    rng = random.Random(seed)   # the same seed every time, so a replayed run gives the same numbers
    m = MATERIALS[design.material]
    L = req.length_mm / 1000
    draws, margins = [], []
    for _ in range(samples):
        x = {k: rng.gauss(0, 1) for k in INPUTS}
        load = req.load_n * (1 + SCATTER["load"] * x["load"])
        h = max(design.thickness_mm + SCATTER["thickness_mm"] * x["thickness_mm"], 0.1) / 1000
        b = max(design.width_mm + SCATTER["width_mm"] * x["width_mm"], 0.1) / 1000
        strength = 1 + SCATTER["strength"] * x["strength"]
        E = m.youngs_modulus * (1 + SCATTER["modulus"] * x["modulus"])
        margins.append(governing_margin(load, b, h, L, m.yield_strength * strength, m.ultimate_strength * strength, E, req))
        draws.append(x)
    failures = sum(g < 0 for g in margins)
    ordered = sorted(margins)

    def corr(values):
        mv, mg = sum(values) / samples, sum(margins) / samples
        cov = sum((v - mv) * (g - mg) for v, g in zip(values, margins))
        sv = math.sqrt(sum((v - mv) ** 2 for v in values))
        sg = math.sqrt(sum((g - mg) ** 2 for g in margins))
        return cov / (sv * sg) if sv and sg else 0.0

    sensitivity = {k: round(corr([d[k] for d in draws]), 3) for k in INPUTS}
    lo, hi = ordered[0], ordered[-1]
    bins = 30
    width = (hi - lo) / bins or 1
    histogram = [0] * bins
    for g in margins:
        histogram[min(int((g - lo) / width), bins - 1)] += 1
    return {"design": design.model_dump(), "samples": samples, "failures": failures, "p_failure": failures / samples,
            "p5": round(ordered[int(0.05 * samples)], 3), "p50": round(ordered[samples // 2], 3),
            "histogram": {"lo": lo, "hi": hi, "counts": histogram}, "sensitivity": sensitivity, "scatter": SCATTER}


def tradeoffs(verified: list[dict], req: Requirement, qty: int = 10) -> list[dict]:
    """Mass, cost, worst-case margin and probability of failure for every verified design."""
    from .suppliers import estimate
    rows = []
    for v in verified:
        d = Design(**v["design"])
        c = check(d, req)
        mc = monte_carlo(d, req, samples=800)
        rows.append({"id": v["id"], "design": v["design"], "mass_g": round(c.mass_g, 1),
                     "cost_usd": estimate(REFERENCE_SHOP, d, req, qty)["price_each_usd"],
                     "margin_worst": round(c.worst_case.margin_yield, 3), "p_failure": mc["p_failure"]})
    return rows
