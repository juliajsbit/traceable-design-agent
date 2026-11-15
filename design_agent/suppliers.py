"""Supplier Desk: send a request for quote to every approved vendor at once, check the answers, pick one.

    rfq -> vendor agent A --+
        -> vendor agent B --+-> procurement
        -> vendor agent C --+
        -> vendor agent D --+

Each vendor is an agent with two tools scoped to its own shop: check_stock and estimate_job.
The numbers come from a plain cost model; the agent decides bid or no-bid and writes the quote.
Then code checks every quote: is the vendor eligible (AS9100, DFARS melt, material), and does the
quoted price match the shop's own estimate? Ranking is code too; a model only writes the memo.
"""
import json
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel

from .physics import MATERIALS
from .schemas import Design, Requirement

VENDORS = json.loads((Path(__file__).parent / "data" / "approved_vendors.json").read_text())["vendors"]

# Illustrative cost inputs, not market prices
PRICE_PER_KG = {"Al 6061-T6": 9.0, "Ti-6Al-4V": 65.0, "Steel 4130": 7.0}
REMOVAL_CM3_MIN = {"Al 6061-T6": 80.0, "Ti-6Al-4V": 10.0, "Steel 4130": 20.0}
SETUP_H, FEATURES_H, INSPECT_H, FAI_USD = 1.5, 0.3, 0.15, 250.0
PLATE_ORDER_DAYS = 10
PRICE_TOLERANCE = 0.05   # a quote more than 5% off the shop's own estimate is flagged
WEIGHTS = {"price": 0.6, "lead_time": 0.4}


class Quote(BaseModel):
    bid: bool
    price_each_usd: float | None = None
    lead_time_days: int | None = None
    notes: str = ""
    no_bid_reasons: list[str] = []


def estimate(vendor: dict, design: Design, req: Requirement, qty: int) -> dict:
    """The shop's own cost model: material for the blank, machining time, inspection, margin."""
    m, t = design.material, design.thickness_mm
    blank_cm3 = (design.width_mm + 10) * (req.length_mm + 10) * t / 1000
    part_cm3 = design.width_mm * req.length_mm * t / 1000
    material_usd = blank_cm3 / 1e6 * MATERIALS[m].density * PRICE_PER_KG[m]   # cm3 -> m3, times kg/m3 -> kg
    hours = SETUP_H / qty + (blank_cm3 - part_cm3) / REMOVAL_CM3_MIN[m] / 60 + FEATURES_H + INSPECT_H
    fai = FAI_USD / qty if "AS9100D" in vendor["certifications"] else 0.0   # first article inspection report
    each = (material_usd + hours * vendor["rate_usd_h"] + fai) * (1 + vendor["margin"])
    in_stock = t in vendor["stock"].get(m, [])
    lead = vendor["base_lead_days"] + round(10 * vendor["shop_load"]) + (0 if in_stock else PLATE_ORDER_DAYS)
    return {"material_usd": round(material_usd, 2), "machine_hours": round(hours, 2), "fai_usd": round(fai, 2),
            "price_each_usd": round(each, 2), "lead_time_days": lead, "plate_in_stock": in_stock}


def eligibility(vendor: dict, design: Design, needs: dict) -> list[str]:
    """Reasons this vendor cannot take the job. Empty means eligible."""
    reasons = []
    if design.material not in vendor["stock"]:
        reasons.append(f"does not work with {design.material}")
    if needs.get("as9100") and "AS9100D" not in vendor["certifications"]:
        reasons.append("no AS9100 quality system")
    if needs.get("dfars_melt") and not vendor["dfars_melt"]:
        reasons.append("cannot certify a DFARS compliant melt source")
    return reasons


def vendor_tools(vendor: dict, design: Design, req: Requirement):
    @tool
    def check_stock(material: str, thickness_mm: float) -> str:
        """Is this plate in stock at your shop? Also returns your certifications and melt-source status."""
        sizes = vendor["stock"].get(material)
        stock = ("material not offered" if sizes is None else
                 "in stock" if thickness_mm in sizes else f"not stocked, about {PLATE_ORDER_DAYS} days to order")
        return json.dumps({"stock": stock, "certifications": vendor["certifications"],
                           "dfars_compliant_melt": vendor["dfars_melt"]})

    @tool
    def estimate_job(quantity: int) -> str:
        """Your shop's cost estimate for the requested part: material, machine hours, price each and lead time."""
        return json.dumps(estimate(vendor, design, req, quantity))

    return [check_stock, estimate_job]


def rfq_text(design: Design, req: Requirement, qty: int, needs: dict) -> str:
    must = ["AS9100 quality system"] if needs.get("as9100") else []
    if needs.get("dfars_melt"):
        must.append("DFARS 252.225-7009 compliant melt source")
    return (f"Request for quote: {qty} brackets, {design.material}, {design.width_mm:g} x {design.thickness_mm:g} x "
            f"{req.length_mm:g} mm, machined from {design.thickness_mm:g} mm plate, two mounting holes. "
            f"Buyer requirements: {', '.join(must) or 'none'}.")


def ask_vendor(vendor: dict, design: Design, req: Requirement, qty: int, needs: dict, llm, emit, track=lambda message: None) -> dict:
    """One vendor agent: a small tool loop, then a structured quote. Returns the quote plus the code checks."""
    tools = vendor_tools(vendor, design, req)
    by_name = {t.name: t for t in tools}
    agent = llm.bind_tools(tools)
    messages = [HumanMessage(
        f"You are the quoting desk at {vendor['name']} in {vendor['location']}.\n{rfq_text(design, req, qty, needs)}\n"
        "Use your tools to check stock and estimate the job. Bid only if you can meet every buyer requirement; "
        "otherwise do not bid and say why. Quote the price from your estimate.")]
    for _ in range(6):
        reply = agent.invoke(messages)
        track(reply)
        messages.append(reply)
        emit({"type": "vendor_step", "vendor": vendor["id"], "tools": [c["name"] for c in reply.tool_calls]})
        if not reply.tool_calls:
            break
        for call in reply.tool_calls:
            messages.append(by_name[call["name"]].invoke(call))   # returns a ToolMessage with the call id
    messages.append(HumanMessage("Give your final quote now."))
    for _ in range(2):  # the history has tool calls, so this uses a tool for the schema; retry once if it comes back malformed
        result = llm.with_structured_output(Quote, include_raw=True).invoke(messages)
        track(result["raw"])
        if result["parsed"]:
            break
    quote = result["parsed"] or Quote(bid=False, no_bid_reasons=["no quote returned"])

    est = estimate(vendor, design, req, qty)
    blockers = eligibility(vendor, design, needs)
    flags = []
    if quote.bid and blockers:
        flags.append("bid rejected: " + "; ".join(blockers))
    if quote.bid and quote.price_each_usd and abs(quote.price_each_usd / est["price_each_usd"] - 1) > PRICE_TOLERANCE:
        flags.append(f"price {quote.price_each_usd:.2f} does not match the shop estimate {est['price_each_usd']:.2f}")
    return {"vendor": vendor["id"], "name": vendor["name"], "location": vendor["location"],
            "certifications": vendor["certifications"], "dfars_melt": vendor["dfars_melt"],
            "quote": quote.model_dump(), "estimate": est, "eligible": not blockers, "blockers": blockers, "flags": flags}


def rank(results: list[dict], qty: int) -> list[dict]:
    """Score eligible, unflagged bids: lower price and shorter lead time are better (0 = best on both)."""
    valid = [r for r in results if r["quote"]["bid"] and r["eligible"] and not r["flags"]]
    if not valid:
        return []
    prices = [r["quote"]["price_each_usd"] for r in valid]
    leads = [r["quote"]["lead_time_days"] or r["estimate"]["lead_time_days"] for r in valid]

    def norm(x, xs):
        return 0.0 if max(xs) == min(xs) else (x - min(xs)) / (max(xs) - min(xs))

    ranked = [{"vendor": r["vendor"], "name": r["name"], "price_each_usd": p, "total_usd": round(p * qty, 2),
               "lead_time_days": d, "score": round(WEIGHTS["price"] * norm(p, prices) + WEIGHTS["lead_time"] * norm(d, leads), 3)}
              for r, p, d in zip(valid, prices, leads)]
    return sorted(ranked, key=lambda x: x["score"])


def memo_prompt(design: Design, results: list[dict], ranked: list[dict], qty: int) -> str:
    lines = [f"{r['name']}: bid={r['quote']['bid']}, price each={r['quote']['price_each_usd']}, lead={r['quote']['lead_time_days']} days, "
             f"eligible={r['eligible']} {r['blockers']}, flags={r['flags']}" for r in results]
    top = ranked[0]["name"] if ranked else "none"
    return (f"You are the buyer for {qty} {design.material} brackets. Quotes and code checks:\n" + "\n".join(lines) +
            f"\nThe scoring picked: {top} (weights: price {WEIGHTS['price']}, lead time {WEIGHTS['lead_time']}). "
            "Write a three-sentence recommendation memo: who, why, and why the others were not chosen. Use only these facts.")
