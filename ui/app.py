import json
import os
import re
from typing import Any

import chainlit as cl
import httpx


API_URL = os.getenv("API_URL", "http://api:8000")


async def request_schema() -> dict[str, object]:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(f"{API_URL}/form-schema")
        response.raise_for_status()
        return response.json()


@cl.on_chat_start
async def start() -> None:
    await open_prediction_flow()


@cl.on_message
async def predict(message: cl.Message) -> None:
    if not cl.user_session.get("form_active"):
        await open_prediction_flow()


async def open_prediction_flow() -> None:
    try:
        schema = cl.user_session.get("schema") or await request_schema()
        cl.user_session.set("schema", schema)
    except httpx.HTTPError as error:
        await cl.Message(content=f"The prediction assistant is unavailable: {error}").send()
        return
    cl.user_session.set("form_active", True)
    try:
        while True:
            instance = await request_customer_profile(schema)
            if instance is None:
                await cl.Message(
                    content="The prediction was cancelled. Send any message to open the form again."
                ).send()
                return
            await stream_prediction(instance)
    finally:
        cl.user_session.set("form_active", False)


async def request_customer_profile(
    schema: dict[str, object],
) -> dict[str, object] | None:
    fields = schema.get("fields", [])
    if not isinstance(fields, list) or not all(isinstance(field, dict) for field in fields):
        await cl.Message(content="The training schema is invalid.").send()
        return None
    element = cl.CustomElement(
        name="CustomerProfileForm",
        display="inline",
        props={"fields": [form_field(field) for field in fields]},
    )
    response = await cl.AskElementMessage(
        content="Complete the customer profile to estimate churn risk.",
        element=element,
        timeout=3600,
    ).send()
    if response is None or not response.get("submitted"):
        return None
    try:
        return {
            str(field["name"]): parse_answer(str(response[str(field["name"])]), field)
            for field in fields
        }
    except (KeyError, ValueError) as error:
        await cl.Message(content=f"The submitted profile is invalid: {error}").send()
        return None


def form_field(field: dict[str, object]) -> dict[str, object]:
    name = str(field["name"])
    result: dict[str, object] = {
        "id": name,
        "label": humanize(name),
        "type": field["type"],
        "required": True,
    }
    if field["type"] == "number":
        result.update(
            {
                "minimum": field["minimum"],
                "maximum": field["maximum"],
                "step": number_step(field),
            }
        )
    else:
        result["options"] = field["options"]
    return result


def humanize(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("last Year", "Last Year")


def number_step(field: dict[str, object]) -> str:
    values = (field["minimum"], field["maximum"])
    return "1" if all(float(value).is_integer() for value in values) else "0.01"


def parse_answer(answer: str, field: dict[str, object]) -> object:
    name = str(field["name"])
    if field["type"] == "number":
        try:
            value = float(answer.replace(",", "."))
        except ValueError as error:
            raise ValueError(f"{name} must be a number") from error
        if float(field["minimum"]) <= value <= float(field["maximum"]):
            return value
        raise ValueError(f"{name} must be within the trained range")
    options = {str(option).casefold(): str(option) for option in field["options"]}
    selected = options.get(answer.strip().casefold())
    if selected is None:
        raise ValueError(f"{name} must be one of the listed options")
    return selected


async def stream_prediction(instance: dict[str, object]) -> None:
    votes: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream(
            "POST", f"{API_URL}/predictions/stream", json=instance
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if event["type"] == "model_vote":
                    votes.append(event["content"])
                async with cl.Step(name=event["type"], type="tool") as step:
                    step.output = format_event(event)
                if event["type"] == "decision":
                    await cl.Message(content=render_result(votes, event["content"])).send()
                if event["type"] == "arbitration_error":
                    await cl.Message(content=render_failure(event["content"])).send()


def format_event(event: dict[str, Any]) -> str:
    return f"{event['stage']} · {event['status']}\n{json.dumps(event['content'], indent=2)}"


def render_result(votes: list[dict[str, object]], decision: dict[str, object]) -> str:
    rows = "\n".join(
        f"| {vote['model']} | {vote['label']} | {vote['confidence']} | {vote['latency_ms']} |"
        for vote in votes
    )
    explanation = decision.get("explanation", "Automatic ensemble decision.")
    return (
        "## Final decision\n"
        f"**{decision['label']}** from {decision['source']}\n\n"
        "| Model | Label | Confidence | Latency (ms) |\n"
        "| --- | --- | --- | --- |\n"
        f"{rows}\n\n{explanation}"
    )


def render_failure(content: dict[str, object]) -> str:
    return (
        "## Arbitration unavailable\n"
        f"The ensemble remained split 3–2 and no final label was produced. {content['message']}"
    )
