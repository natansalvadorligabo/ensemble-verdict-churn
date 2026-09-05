from collections import Counter
from dataclasses import dataclass
from typing import Literal

Agreement = Literal["unanimity", "consensus", "arbitration"]


@dataclass(frozen=True)
class ModelVote:
    model: str
    label: str
    confidence: float | None
    latency_ms: float


@dataclass(frozen=True)
class Aggregation:
    agreement: Agreement
    majority_label: str
    tally: dict[str, int]


@dataclass(frozen=True)
class ArbitrationResult:
    label: str
    explanation: str


def aggregate_votes(votes: list[ModelVote]) -> Aggregation:
    if len(votes) != 5:
        raise ValueError("Exactly five model votes are required")
    tally = dict(Counter(vote.label for vote in votes))
    majority_label, majority_count = max(tally.items(), key=lambda item: item[1])
    agreement: Agreement = (
        "unanimity"
        if majority_count == 5
        else "consensus"
        if majority_count == 4
        else "arbitration"
    )
    return Aggregation(agreement=agreement, majority_label=majority_label, tally=tally)
