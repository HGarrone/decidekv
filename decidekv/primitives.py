"""Jev / OpenJev-compatible question and answer types (no dependencies)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Union


@dataclass
class Choice:
    """Pick one option. `options` is a list of strings, a dict {key: description}, or a prebuilt `Index`."""
    instructions: str
    options: Any = None
    criteria: dict[str, str] | None = None      # Jev-style alias for options={key: description}
    type: str = "choice"


@dataclass
class Score:
    """Grade on an ordered scale; `criteria` lists the levels from lowest (0) to highest."""
    instructions: str
    criteria: list[str] = field(default_factory=list)
    type: str = "score"


@dataclass
class Noul:
    """Yes/no judgement; the answer is P(true). `criteria` may describe {"true": ..., "false": ...}."""
    instructions: str
    criteria: dict[str, str] | None = None
    type: str = "noul"


Question = Union[Choice, Score, Noul]


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: dict[str, float]
    confidence: float
    type: str = "choice"


@dataclass
class ScoreAnswer:
    score: float
    probabilities: dict[str, float]
    confidence: float
    legend: dict[str, str]
    type: str = "score"


@dataclass
class NoulAnswer:
    noul: float
    type: str = "noul"


@dataclass
class Usage:
    input_tokens: int = 0          # prompt tokens actually computed
    cached_tokens: int = 0         # prompt tokens served from the KV cache
    output_tokens: int = 0


@dataclass
class Response:
    model: str
    answers: dict[str, Any]
    usage: Usage
    latency_ms: float
    # questions the trained catalogue passed to the LLM (cascade)
    escalated: list[str] = field(default_factory=list)
    # questions answered from the semantic cache (no LLM call)
    cached: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def softmax(logps: dict[str, float]) -> dict[str, float]:
    if not logps:
        return {}
    m = max(logps.values())
    e = {k: math.exp(v - m) for k, v in logps.items()}
    z = sum(e.values())
    return {k: v / z for k, v in e.items()}


def confidence(probs: list[float]) -> float:
    """1 - normalised entropy (same definition as OpenJev)."""
    n = len(probs)
    if n <= 1:
        return 1.0
    h = -sum(p * math.log(p) for p in probs if p > 0)
    return max(0.0, min(1.0, 1.0 - h / math.log(n)))


def answer_confidence(answer) -> float:
    """Confidence of an answer: the `confidence` field of a choice or a score; for a noul, the confidence of the
    distribution (P(true), P(false))."""
    if answer.type == "noul":
        return confidence([answer.noul, 1.0 - answer.noul])
    else:
        return float(answer.confidence)
