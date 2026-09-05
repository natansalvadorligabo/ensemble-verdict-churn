import json
import os
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from app.domain import ModelVote, aggregate_votes
from app.services import (
    Arbiter,
    ArtifactEnsemble,
    Ensemble,
    OllamaArbiter,
    load_form_schema,
    validate_instance,
)


def create_app(ensemble: Ensemble | None = None, arbiter: Arbiter | None = None) -> FastAPI:
    app = FastAPI(title="Ensemble Verdict Churn API")
    app.state.ensemble = ensemble
    app.state.arbiter = arbiter or OllamaArbiter()

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/form-schema")
    async def form_schema() -> dict[str, Any]:
        metadata_path = Path(os.getenv("ARTIFACT_DIRECTORY", "artifacts")) / "form_schema.json"
        if not metadata_path.exists():
            raise HTTPException(503, "Training artifacts are not available")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    @app.post("/predictions/stream")
    async def prediction_stream(instance: dict[str, object]) -> StreamingResponse:
        if not instance:
            raise HTTPException(422, "A customer instance is required")
        artifact_directory = Path(os.getenv("ARTIFACT_DIRECTORY", "artifacts"))
        schema_path = artifact_directory / "form_schema.json"
        if schema_path.exists():
            try:
                instance = validate_instance(instance, load_form_schema(schema_path))
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
        active_ensemble = app.state.ensemble
        if active_ensemble is None:
            try:
                active_ensemble = ArtifactEnsemble(artifact_directory)
            except FileNotFoundError as error:
                raise HTTPException(503, "Training artifacts are not available") from error
            app.state.ensemble = active_ensemble
        return StreamingResponse(
            event_stream(active_ensemble, app.state.arbiter, instance),
            media_type="text/event-stream",
        )

    return app


async def event_stream(
    ensemble: Ensemble, arbiter: Arbiter, instance: dict[str, object]
) -> AsyncIterator[str]:
    yield encode_event("request_started", "request", "started", {"instance": instance})
    yield encode_event("ensemble_started", "ensemble", "started", {})
    votes: list[ModelVote] = []
    async for vote in ensemble.predict(instance):
        votes.append(vote)
        yield encode_event("model_vote", "classification", "completed", vote.__dict__)
    aggregation = aggregate_votes(votes)
    yield encode_event(
        "aggregation",
        "ensemble",
        "completed",
        {
            "agreement": aggregation.agreement,
            "tally": aggregation.tally,
            "majority_label": aggregation.majority_label,
        },
    )
    if aggregation.agreement != "arbitration":
        yield encode_event(
            "decision",
            "ensemble",
            "completed",
            {
                "label": aggregation.majority_label,
                "source": "ensemble",
            },
        )
        return
    yield encode_event("arbitration_started", "arbitration", "started", {})
    try:
        result = await arbiter.arbitrate(instance, votes, set(aggregation.tally))
        yield encode_event("arbitration", "arbitration", "completed", result.__dict__)
        yield encode_event(
            "decision",
            "arbitration",
            "completed",
            {
                "label": result.label,
                "source": "arbiter",
                "contradicts_simple_majority": result.label != aggregation.majority_label,
                "explanation": result.explanation,
            },
        )
    except Exception as error:
        yield encode_event("arbitration_error", "arbitration", "failed", {"message": str(error)})


def encode_event(event_type: str, stage: str, status: str, content: dict[str, object]) -> str:
    event = {
        "type": event_type,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": stage,
        "status": status,
        "content": content,
    }
    return f"data: {json.dumps(event)}\n\n"


app = create_app()
