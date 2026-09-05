import json
from collections.abc import AsyncIterator

import httpx
import pytest

from app.api import create_app
from app.domain import ArbitrationResult, ModelVote


class StubEnsemble:
    def __init__(self, labels: list[str]) -> None:
        self.labels = labels

    async def predict(self, instance: dict[str, object]) -> AsyncIterator[ModelVote]:
        for index, label in enumerate(self.labels):
            yield ModelVote(
                model=("KNN", "SVM-RBF", "Random Forest", "XGBoost", "Naive Bayes")[index],
                label=label,
                confidence=0.9,
                latency_ms=1.0,
            )


class StubArbiter:
    def __init__(self, result: ArbitrationResult | Exception) -> None:
        self.result = result
        self.calls = 0

    async def arbitrate(
        self, instance: dict[str, object], votes: list[ModelVote], labels: set[str]
    ) -> ArbitrationResult:
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


async def stream_events(client: httpx.AsyncClient) -> list[dict[str, object]]:
    response = await client.post(
        "/predictions/stream", json={"Tenure": 4, "PreferredLoginDevice": "Mobile Phone"}
    )
    assert response.status_code == 200
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_unanimous_prediction_streams_all_votes_without_arbitration() -> None:
    arbiter = StubArbiter(ArbitrationResult(label="No", explanation="Not used"))
    app = create_app(StubEnsemble(["No"] * 5), arbiter)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events = await stream_events(client)

    assert [event["type"] for event in events] == [
        "request_started",
        "ensemble_started",
        "model_vote",
        "model_vote",
        "model_vote",
        "model_vote",
        "model_vote",
        "aggregation",
        "decision",
    ]
    assert events[-1]["content"]["label"] == "No"
    assert events[-1]["content"]["source"] == "ensemble"
    assert arbiter.calls == 0


@pytest.mark.asyncio
async def test_consensus_prediction_does_not_call_the_arbiter() -> None:
    arbiter = StubArbiter(ArbitrationResult(label="No", explanation="Not used"))
    app = create_app(StubEnsemble(["No", "No", "No", "No", "Yes"]), arbiter)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events = await stream_events(client)

    assert events[-2]["content"]["agreement"] == "consensus"
    assert events[-1]["content"]["label"] == "No"
    assert arbiter.calls == 0


@pytest.mark.asyncio
async def test_split_prediction_calls_arbiter_and_emits_its_decision() -> None:
    arbiter = StubArbiter(
        ArbitrationResult(label="Yes", explanation="The risk indicators are stronger.")
    )
    app = create_app(StubEnsemble(["No", "No", "No", "Yes", "Yes"]), arbiter)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events = await stream_events(client)

    assert [event["type"] for event in events][-3:] == [
        "arbitration_started",
        "arbitration",
        "decision",
    ]
    assert events[-1]["content"] == {
        "label": "Yes",
        "source": "arbiter",
        "contradicts_simple_majority": True,
        "explanation": "The risk indicators are stronger.",
    }
    assert arbiter.calls == 1


@pytest.mark.asyncio
async def test_arbitration_failure_is_explicit_and_has_no_final_label() -> None:
    arbiter = StubArbiter(RuntimeError("Ollama is unavailable"))
    app = create_app(StubEnsemble(["No", "No", "No", "Yes", "Yes"]), arbiter)
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        events = await stream_events(client)

    assert events[-1]["type"] == "arbitration_error"
    assert events[-1]["content"] == {"message": "Ollama is unavailable"}
