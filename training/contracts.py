"""Strict text contract between the language model and intent arithmetic."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

VARIABLES = {"ri", "ro", "alpha", "Wf", "Lc"}
TOKENS = {
    "increase_strong",
    "increase",
    "increase_slight",
    "hold",
    "decrease_slight",
    "decrease",
    "decrease_strong",
}


class IntentParseError(ValueError):
    """The completion does not contain one executable qualitative intent."""


@dataclass(frozen=True)
class ParsedCompletion:
    reasoning: str
    intent: dict[str, str]
    stop: bool = False


def _extract_json(text: str) -> tuple[str, int, int]:
    tagged = re.search(r"<intent>\s*(\{.*?\})\s*</intent>", text, re.S)
    if tagged:
        return tagged.group(1), tagged.start(1), tagged.end(1)

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    if fenced:
        return fenced.group(1), fenced.start(1), fenced.end(1)

    decoder = json.JSONDecoder()
    candidates: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\{", text):
        try:
            _, length = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        candidates.append(
            (text[match.start():match.start() + length], match.start(), match.start() + length)
        )
    if not candidates:
        raise IntentParseError("completion contains no valid JSON object")
    return candidates[-1]


def _reasoning_from(text: str, start: int) -> str:
    reasoning_match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.S)
    if reasoning_match:
        return reasoning_match.group(1).strip()
    return text[:start].replace("```json", "").replace("```", "").strip()


def parse_intent_completion(text: str, *, allow_stop: bool = False) -> ParsedCompletion:
    """Extract and validate one intent from tagged, fenced, or bare JSON."""
    raw_json, start, _ = _extract_json(text)
    try:
        value = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise IntentParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise IntentParseError("intent JSON must be an object")
    if not value:
        raise IntentParseError("intent must contain at least one variable")

    if allow_stop and str(value.get("action", "")).lower() == "stop":
        return ParsedCompletion(reasoning=_reasoning_from(text, start), intent={}, stop=True)

    unknown = set(value) - VARIABLES
    if unknown:
        raise IntentParseError(f"unknown variable(s): {sorted(unknown)}")
    for variable, token in value.items():
        if not isinstance(token, str) or token not in TOKENS:
            raise IntentParseError(f"invalid token for {variable}: {token!r}")

    active = sum(token != "hold" for token in value.values())
    if active == 0:
        if allow_stop:
            return ParsedCompletion(
                reasoning=_reasoning_from(text, start), intent={}, stop=True
            )
        raise IntentParseError("all-hold intent cannot change the circuit")
    if active > 2:
        raise IntentParseError("intent may change at most two variables")

    return ParsedCompletion(reasoning=_reasoning_from(text, start), intent=dict(value))

