import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol

import httpx
import joblib
import pandas as pd

from app.domain import ArbitrationResult, ModelVote

MODEL_NAMES = ("KNN", "SVM-RBF", "Random Forest", "XGBoost", "Naive Bayes")


class Ensemble(Protocol):
    async def predict(self, instance: dict[str, object]) -> AsyncIterator[ModelVote]: ...


class Arbiter(Protocol):
    async def arbitrate(
        self, instance: dict[str, object], votes: list[ModelVote], labels: set[str]
    ) -> ArbitrationResult: ...


class ProfileExtractor(Protocol):
    async def extract(
        self,
        message: str,
        current_profile: dict[str, object],
        form_schema: dict[str, object],
    ) -> dict[str, object]: ...


class AnalysisExplainer(Protocol):
    async def explain(self, question: str, analysis: dict[str, object]) -> str: ...

    def stream(self, question: str, analysis: dict[str, object]) -> AsyncIterator[str]: ...


class ConversationInterpreter(Protocol):
    async def interpret(self, message: str, context: dict[str, object]) -> str: ...


class ArtifactEnsemble:
    def __init__(self, artifact_directory: Path) -> None:
        self.artifact_directory = artifact_directory
        self.models = {
            name: joblib.load(artifact_directory / f"{artifact_filename(name)}.pkl")
            for name in MODEL_NAMES
        }

    async def predict(self, instance: dict[str, object]) -> AsyncIterator[ModelVote]:
        features = pd.DataFrame([instance])
        for name, model in self.models.items():
            started_at = time.perf_counter()
            label = str(model.predict(features)[0])
            confidence = self._confidence(model, features, label)
            latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
            yield ModelVote(name, label, confidence, latency_ms)

    def _confidence(self, model: Any, features: pd.DataFrame, label: str) -> float | None:
        if not hasattr(model, "predict_proba"):
            return None
        probabilities = model.predict_proba(features)[0]
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
            "response_schema": arbitration_schema(labels),
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": arbitration_schema(labels),
                    "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": json.dumps(prompt)}],
                },
            )
            response.raise_for_status()
        content = response.json()["message"]["content"]
        return normalize_arbitration_response(content, labels)


class OllamaProfileExtractor:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

    async def extract(
        self,
        message: str,
        current_profile: dict[str, object],
        form_schema: dict[str, object],
    ) -> dict[str, object]:
        output_schema = profile_extraction_schema(form_schema)
        prompt = {
            "instruction": (
                "Extract customer attributes from the user's message in any language. "
                "Use only the supplied field names and categorical values. "
                "Return null for every value that was not explicitly provided or corrected."
            ),
            "current_profile": current_profile,
            "user_message": message,
            "field_schema": form_schema,
            "response_schema": output_schema,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": output_schema,
                    "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": json.dumps(prompt)}],
                },
            )
            response.raise_for_status()
        content = json.loads(response.json()["message"]["content"])
        extraction = normalize_profile_extraction(content, form_schema)
        merged = {**current_profile, **extraction["profile"]}
        complete_payload = {
            "profile": {
                str(field["name"]): merged.get(str(field["name"]))
                for field in form_schema["fields"]
            }
        }
        return normalize_profile_extraction(complete_payload, form_schema)


class OllamaAnalysisExplainer:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

    async def explain(self, question: str, analysis: dict[str, object]) -> str:
        output_schema = {
            "type": "object",
            "properties": {"answer": {"type": "string", "minLength": 1}},
            "required": ["answer"],
            "additionalProperties": False,
        }
        prompt = {
            "instruction": (
                "Answer the follow-up question in the user's language using only the recorded "
                "customer profile, model votes, confidences, and final decision. Label 1 means "
                "predicted churn and label 0 means predicted retention. Explain associations, "
                "not causal claims, and acknowledge disagreement when present."
            ),
            "analysis": analysis,
            "question": question,
            "response_schema": output_schema,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": output_schema,
                    "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": json.dumps(prompt)}],
                },
            )
            response.raise_for_status()
        payload = json.loads(response.json()["message"]["content"])
        answer = str(payload.get("answer", "")).strip()
        if not answer:
            raise ValueError("The analysis explanation is empty")
        return answer

    async def stream(
        self, question: str, analysis: dict[str, object]
    ) -> AsyncIterator[str]:
        prompt = {
            "instruction": (
                "Answer the follow-up question in the user's language using only the recorded "
                "customer profile, model votes, confidences, and final decision. Label 1 means "
                "predicted churn and label 0 means predicted retention. Explain associations, "
                "not causal claims, and acknowledge disagreement when present. A confidence value "
                "supports only that model's own label, never the opposite label. Return plain text."
            ),
            "analysis": analysis,
            "question": question,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": True,
                    "think": False,
                    "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": json.dumps(prompt)}],
                },
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    token = str(json.loads(line).get("message", {}).get("content", ""))
                    if token:
                        yield token


class OllamaConversationInterpreter:
    def __init__(self, base_url: str | None = None, model: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://ollama:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:1.7b")

    async def interpret(self, message: str, context: dict[str, object]) -> str:
        if context.get("profile_awaiting_confirmation") is True:
            intents = ["confirm_profile", "cancel_profile", "update_profile"]
        elif context.get("profile_in_progress") is True:
            intents = ["cancel_profile", "update_profile"]
        elif context.get("completed_prediction_available") is True:
            intents = ["ask_about_result", "describe_customer"]
        else:
            intents = ["describe_customer"]
        output_schema = {
            "type": "object",
            "properties": {"intent": {"type": "string", "enum": intents}},
            "required": ["intent"],
            "additionalProperties": False,
        }
        definitions = {
            "confirm_profile": "The user approves the displayed profile and wants prediction to run.",
            "cancel_profile": "The user withdraws, gives up, stops, or does not want the prediction.",
            "update_profile": "The user corrects or adds customer attributes before prediction.",
            "ask_about_result": "The user asks about the meaning, evidence, models, or completed result.",
            "describe_customer": "The user provides attributes for a different customer.",
        }
        prompt = {
            "instruction": (
                "Infer the user's conversational intent in any language. Confirm or cancel only "
                "when a profile is awaiting confirmation. Expressions granting permission to "
                "proceed mean confirm_profile. Treat profile corrections as updates. "
                "When a completed prediction exists, distinguish questions about that result from "
                "descriptions of another customer. Use describe_customer for customer details."
            ),
            "available_intents": {intent: definitions[intent] for intent in intents},
            "examples": [
                {"message": "Pode prosseguir com esses dados", "intent": "confirm_profile"},
                {"message": "Forget it, I do not want to continue", "intent": "cancel_profile"},
                {"message": "Deixa pra lá", "intent": "cancel_profile"},
                {"message": "Change the distance to 20 km", "intent": "update_profile"},
                {"message": "Why did the models disagree?", "intent": "ask_about_result"},
                {"message": "Here is another customer", "intent": "describe_customer"},
            ],
            "conversation_context": context,
            "user_message": message,
            "response_schema": output_schema,
        }
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "stream": False,
                    "think": False,
                    "format": output_schema,
                    "options": {"temperature": 0},
                    "messages": [{"role": "user", "content": json.dumps(prompt)}],
                },
            )
            response.raise_for_status()
        intent = str(json.loads(response.json()["message"]["content"]).get("intent", ""))
        if intent not in intents:
            raise ValueError("The conversation intent is invalid")
        return intent


def arbitration_schema(labels: set[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": sorted(labels)},
            "explanation": {"type": "string", "minLength": 1},
        },
        "required": ["label", "explanation"],
        "additionalProperties": False,
    }


def profile_extraction_schema(form_schema: dict[str, object]) -> dict[str, object]:
    fields = form_schema.get("fields")
    if not isinstance(fields, list):
        raise ValueError("The artifact form schema is invalid")
    properties: dict[str, object] = {}
    required: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError("The artifact form schema is invalid")
        name = str(field["name"])
        required.append(name)
        if field["type"] == "number":
            value_schema = {
                "type": "number",
                "minimum": field["minimum"],
                "maximum": field["maximum"],
            }
        else:
            value_schema = {"type": "string", "enum": field["options"]}
        properties[name] = {"anyOf": [value_schema, {"type": "null"}]}
    return {
        "type": "object",
        "properties": {
            "profile": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }
        },
        "required": ["profile"],
        "additionalProperties": False,
    }


def normalize_profile_extraction(
    content: str | dict[str, object], form_schema: dict[str, object]
) -> dict[str, object]:
    payload = json.loads(content) if isinstance(content, str) else content
    profile = payload.get("profile")
    fields = form_schema.get("fields")
    if not isinstance(profile, dict) or not isinstance(fields, list):
        raise ValueError("The extracted customer profile is invalid")
    validated: dict[str, object] = {}
    missing: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            raise ValueError("The artifact form schema is invalid")
        name = str(field["name"])
        value = profile.get(name)
        if value is None:
            missing.append(name)
            continue
        validated[name] = validate_field_value(name, value, field)
    return {"profile": validated, "missing_fields": missing}


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
        validated[name] = validate_field_value(name, instance[name], field)
    return validated


def validate_field_value(name: str, value: object, field: dict[str, object]) -> object:
    if field["type"] == "number":
        try:
            numeric_value = float(str(value))
        except ValueError as error:
            raise ValueError(f"{name} must be numeric") from error
        if numeric_value < float(field["minimum"]) or numeric_value > float(field["maximum"]):
            raise ValueError(f"{name} is outside the trained range")
        return numeric_value
    if str(value) in field.get("options", []):
        return str(value)
    raise ValueError(f"{name} is not a trained category")
