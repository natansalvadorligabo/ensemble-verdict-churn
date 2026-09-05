import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import httpx
import joblib

from app.domain import ArbitrationResult, ModelVote

MODEL_NAMES = ("KNN", "SVM-RBF", "Random Forest", "XGBoost", "Naive Bayes")


class Ensemble(Protocol):
    async def predict(self, instance: dict[str, object]) -> AsyncIterator[ModelVote]: ...


class Arbiter(Protocol):
    async def arbitrate(
        self, instance: dict[str, object], votes: list[ModelVote], labels: set[str]
    ) -> ArbitrationResult: ...


class ArtifactEnsemble:
    def __init__(self, artifact_directory: Path) -> None:
        self.artifact_directory = artifact_directory
        self.models = {
            name: joblib.load(artifact_directory / f"{artifact_filename(name)}.pkl")
            for name in MODEL_NAMES
        }

    async def predict(self, instance: dict[str, object]) -> AsyncIterator[ModelVote]:
        for name, model in self.models.items():
            started_at = time.perf_counter()
            label = str(model.predict([instance])[0])
            confidence = self._confidence(model, instance, label)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            yield ModelVote(name, label, confidence, latency_ms)

    def _confidence(self, model: Any, instance: dict[str, object], label: str) -> float | None:
        if not hasattr(model, "predict_proba"):
            return None
        probabilities = model.predict_proba([instance])[0]
        classes = [str(item) for item in model.classes_]
        return round(float(probabilities[classes.index(label)]), 4)


class OllamaArbiter:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

    async def arbitrate(
        self, instance: dict[str, object], votes: list[ModelVote], labels: set[str]
    ) -> ArbitrationResult:
        prompt = {
            "allowed_labels": sorted(labels),
            "instance": instance,
            "votes": [vote.__dict__ for vote in votes],
            "instruction": "Choose one allowed label and give a short evidence-based explanation.",
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "format": "json",
                    "messages": [{"role": "user", "content": json.dumps(prompt)}],
                },
            )
            response.raise_for_status()
        content = response.json()["message"]["content"]
        return normalize_arbitration_response(content, labels)


def normalize_arbitration_response(
    content: str | dict[str, object], labels: set[str]
) -> ArbitrationResult:
    payload = json.loads(content) if isinstance(content, str) else content
    label = str(payload.get("label", ""))
    explanation = str(payload.get("explanation", "")).strip()
    if label not in labels:
        raise ValueError("Arbiter returned a label outside the allowed classes")
    if not explanation:
        raise ValueError("Arbiter returned no explanation")
    return ArbitrationResult(label=label, explanation=explanation)


def artifact_filename(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def load_form_schema(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_instance(instance: dict[str, object], schema: dict[str, object]) -> dict[str, object]:
    fields = schema.get("fields")
    if not isinstance(fields, list):
        raise ValueError("The artifact form schema is invalid")
    expected_names = {str(field["name"]) for field in fields if isinstance(field, dict)}
    if set(instance) != expected_names:
        raise ValueError("The customer instance must contain exactly the trained feature fields")
    validated: dict[str, object] = {}
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError("The artifact form schema is invalid")
        name = str(field["name"])
        value = instance[name]
        if field["type"] == "number":
            try:
                numeric_value = float(str(value))
            except ValueError as error:
                raise ValueError(f"{name} must be numeric") from error
            if numeric_value < float(field["minimum"]) or numeric_value > float(field["maximum"]):
                raise ValueError(f"{name} is outside the trained range")
            validated[name] = numeric_value
        elif str(value) in field.get("options", []):
            validated[name] = str(value)
        else:
            raise ValueError(f"{name} is not a trained category")
    return validated
