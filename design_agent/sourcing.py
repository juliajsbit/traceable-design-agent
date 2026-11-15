"""Finds real shops that could make the selected part, using the model's web search tool.

A vendor is kept only if its link came back in the search results, so a name the model
made up never reaches the screen.
"""
import json
import re
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError

from .proposers import FALLBACK, PRICE_PER_MTOK
from .schemas import Design, Requirement

SEARCH_PRICE = 0.01  # USD per web search
# web_search_20260209 filters results with code before they reach the model; Haiku gets the basic version
BASIC_SEARCH = {"claude-haiku-5-5"}


class Vendor(BaseModel):
    name: str
    location: str
    url: str
    capabilities: str
    certifications: str


def _domain(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return host.removeprefix("www.")


def prompt(design: Design, req: Requirement) -> str:
    return (
        f"Find up to 4 US companies that could CNC machine this aerospace part: a {design.material} plate bracket, "
        f"{design.width_mm:g} x {design.thickness_mm:g} x {req.length_mm:g} mm, cut from {design.thickness_mm:g} mm plate. "
        "Prefer machine shops in Southern California that state an AS9100 certification and work with this material. "
        "Only list companies you found in the search results, and use the exact URL from the result. "
        "For certifications, write only what their site states, or 'not stated'. "
        'End your answer with a JSON array inside <vendors></vendors> tags, each item with the keys '
        '"name", "location", "url", "capabilities", "certifications".'
    )


def find_vendors(design: Design, req: Requirement, model: str, client=None, max_searches: int = 5) -> dict:
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    tool = {"type": "web_search_20250305" if model in BASIC_SEARCH else "web_search_20260209", "name": "web_search",
            "max_uses": max_searches,
            "user_location": {"type": "approximate", "city": "Long Beach", "region": "California", "country": "US"}}
    extra = {} if model in BASIC_SEARCH else FALLBACK  # Haiku has no server-side fallback
    messages = [{"role": "user", "content": prompt(design, req)}]
    responses = []
    for _ in range(4):
        response = client.beta.messages.create(model=model, max_tokens=8000, tools=[tool], messages=messages, **extra)
        responses.append(response)
        if response.stop_reason == "refusal":
            raise RuntimeError("model refused the search")
        if response.stop_reason != "pause_turn":  # a long server-side search can pause; send it back to continue
            break
        messages.append({"role": "assistant", "content": response.content})

    sources, queries = {}, []
    for response in responses:
        for block in response.content:
            if block.type == "server_tool_use" and block.name == "web_search":
                queries.append(block.input.get("query", ""))
            elif block.type == "web_search_tool_result" and isinstance(block.content, list):  # an error is an object, not a list
                for r in block.content:
                    sources[_domain(r.url)] = r.url

    text = "".join(b.text for b in responses[-1].content if b.type == "text")
    found = re.search(r"<vendors>(.*?)</vendors>", text, re.S)
    try:
        raw = json.loads(found.group(1)) if found else []
    except json.JSONDecodeError:
        raw = []
    kept, dropped = [], 0
    for v in raw:
        try:
            vendor = Vendor(**v)
        except (ValidationError, TypeError):
            dropped += 1
            continue
        d = _domain(vendor.url)
        if d and any(d == s or d.endswith("." + s) or s.endswith("." + d) for s in sources):
            kept.append(vendor.model_dump())
        else:
            dropped += 1

    tokens_in = sum(r.usage.input_tokens for r in responses)
    tokens_out = sum(r.usage.output_tokens for r in responses)
    searches = sum(getattr(getattr(r.usage, "server_tool_use", None), "web_search_requests", 0) or 0 for r in responses)
    price_in, price_out = PRICE_PER_MTOK.get(model, (0, 0))
    return {"vendors": kept, "dropped": dropped, "queries": queries, "sources": len(sources),
            "usage": {"llm_calls": len(responses), "tool_calls": searches, "input_tokens": tokens_in,
                      "output_tokens": tokens_out,
                      "cost_usd": (tokens_in * price_in + tokens_out * price_out) / 1e6 + searches * SEARCH_PRICE}}
