"""Never trust a proposer's numbers. Recompute them and record every disagreement."""
from .physics import check
from .schemas import Check, Finding, Proposal, Record, Requirement

MARGIN_TOLERANCE = 0.10    # a claimed margin more than 0.10 away from the physics is flagged
RELATIVE_TOLERANCE = 0.10  # deflection and mass: more than 10% away is flagged
STRENGTH_AND_STIFFNESS = {"REQ-001", "REQ-002", "REQ-003"}
PHYSICAL = STRENGTH_AND_STIFFNESS | {"REQ-007"}


def _failed(c: Check) -> list[str]:
    return [r.id for r in c.results if not r.passed]


def _label(c: Check) -> str:
    failed = set(_failed(c))
    if failed & PHYSICAL:
        return "fails"
    if "REQ-005" in failed:
        return "not manufacturable"
    if "REQ-006" in failed:
        return "fragile"
    return "verified"


def label(c: Check) -> str:
    """The label for a raw check, including designs outside the envelope (verify() catches those before checking)."""
    return "out of envelope" if "REQ-004" in _failed(c) else _label(c)


def verify(p: Proposal, req: Requirement) -> Record:
    d = p.design
    (wmin, wmax), (tmin, tmax) = req.width_range_mm, req.thickness_range_mm
    if not (wmin <= d.width_mm <= wmax and tmin <= d.thickness_mm <= tmax):
        msg = f"REQ-004: {d.width_mm:g} x {d.thickness_mm:g} mm is outside the envelope"
        return Record(proposal=p, check=None, findings=[Finding(kind="out_of_bounds", message=msg)],
                      status="rejected", label="out of envelope", reason=msg)

    c = check(d, req)
    findings = []

    claimed_ok = p.claimed_margin_yield >= 0 and p.claimed_deflection_mm <= req.max_deflection_mm
    if claimed_ok and set(_failed(c)) & STRENGTH_AND_STIFFNESS:
        why = "; ".join(f"{r.id} {r.value}" for r in c.results if r.id in STRENGTH_AND_STIFFNESS and not r.passed)
        findings.append(Finding(kind="false_pass", message=f"claimed it passes, physics says: {why}"))

    if abs(p.claimed_margin_yield - c.nominal.margin_yield) > MARGIN_TOLERANCE:
        findings.append(Finding(kind="claim_off", message=f"yield margin: claimed {p.claimed_margin_yield:+.2f}, actual {c.nominal.margin_yield:+.2f}"))
    for name, claimed, actual in [("deflection", p.claimed_deflection_mm, c.nominal.deflection_mm),
                                  ("mass", p.claimed_mass_g, c.mass_g)]:
        if abs(claimed - actual) > RELATIVE_TOLERANCE * abs(actual):
            findings.append(Finding(kind="claim_off", message=f"{name}: claimed {claimed:.1f}, actual {actual:.1f}"))

    label = _label(c)
    if label == "verified":
        reason = f"all requirements met; yield MS {c.nominal.margin_yield:+.2f}, worst-case MS {c.worst_case.margin_yield:+.2f}"
    else:
        reason = "; ".join(f"{r.id} {r.title.lower()}: {r.value} (needs {r.limit})" for r in c.results if not r.passed)
    return Record(proposal=p, check=c, findings=findings, status="accepted" if c.passes else "rejected",
                  label=label, reason=reason)


def select(records: list[Record]) -> tuple[str | None, str]:
    """Pick the lightest design whose pass was verified by the physics, never by a claim,
    and say which lighter options were dropped and why, so the choice can be defended."""
    accepted = [r for r in records if r.status == "accepted"]
    if not accepted:
        return None, "no design passed verification"
    best = min(accepted, key=lambda r: r.check.mass_g)
    lighter = sorted((r for r in records if r.check and r.status == "rejected" and r.check.mass_g < best.check.mass_g),
                     key=lambda r: r.check.mass_g)
    reason = f"lightest of {len(accepted)} verified designs at {best.check.mass_g:.0f} g"
    if lighter:
        reason += "; lighter options were rejected: " + ", ".join(
            f"{r.proposal.id} {r.check.mass_g:.0f} g ({r.label})" for r in lighter[:4])
    return best.proposal.id, reason


def feedback(records: list[Record]) -> list[str]:
    """What the proposer is told before the next round."""
    lines = []
    for r in records:
        d = r.proposal.design
        head = f"{d.material} {d.width_mm:g} x {d.thickness_mm:g} mm"
        if r.status == "accepted":
            lines.append(f"{head} passed every requirement at {r.check.mass_g:.0f} g")
        else:
            lines.append(f"{head} was rejected ({r.label}): {r.reason}")
        lines += [f"  your claim was wrong: {f.message}" for f in r.findings if f.kind != "out_of_bounds"]
    return lines
