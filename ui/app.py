import json
import os
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
    try:
        schema = await request_schema()
        cl.user_session.set("schema", schema)
        await cl.Message(
            content="I can guide you through a churn prediction. Send any message when you are ready."
        ).send()
    except httpx.HTTPError as error:
        await cl.Message(content=f"The prediction assistant is unavailable: {error}").send()


@cl.on_message
async def predict(message: cl.Message) -> None:
    schema = cl.user_session.get("schema")
    if schema is None:
        try:
            schema = await request_schema()
            cl.user_session.set("schema", schema)
        except httpx.HTTPError as error:
            await cl.Message(content=f"The prediction assistant is unavailable: {error}").send()
            return
    instance = await collect_instance(schema)
    if instance is not None:
        await stream_prediction(instance)


async def collect_instance(schema: dict[str, object]) -> dict[str, object] | None:
    fields = schema.get("fields", [])
    if not isinstance(fields, list):
        await cl.Message(content="The training schema is invalid.").send()
        return None
    await cl.Message(content="I will ask for one customer attribute at a time.").send()
    instance: dict[str, object] = {}
    for field in fields:
        if not isinstance(field, dict):
            await cl.Message(content="The training schema is invalid.").send()
            return None
        value = await ask_for_value(field)
        if value is None:
            await cl.Message(content="The prediction was cancelled. Send any message to start again.").send()
            return None
        instance[str(field["name"])] = value
    return instance


async def ask_for_value(field: dict[str, object]) -> object | None:
    while True:
        response = await cl.AskUserMessage(content=field_prompt(field), timeout=600).send()
        if response is None:
            return None
        try:
            return parse_answer(str(response["output"]), field)
        except ValueError as error:
            await cl.Message(content=str(error)).send()


def field_prompt(field: dict[str, object]) -> str:
    name = str(field["name"])
    if field["type"] == "number":
        return f"{name}: enter a number from {field['minimum']} to {field['maximum']}."
    options = ", ".join(str(option) for option in field["options"])
    return f"{name}: choose one of: {options}."


def parse_answer(answer: str, field: dict[str, object]) -> object:
    name = str(field["name"])
    if field["type"] == "number":
        try:
            value = float(answer.replace(",", "."))
        except ValueError as error:
            raise ValueError(f"{name} must be a number.") from error
        if float(field["minimum"]) <= value <= float(field["maximum"]):
            return value
        raise ValueError(f"{name} must be within the trained range.")
    options = {str(option).casefold(): str(option) for option in field["options"]}
    selected = options.get(answer.strip().casefold())
    if selected is None:
        raise ValueError(f"{name} must be one of the listed options.")
    return selected


async def stream_prediction(instance: dict[str, object]) -> None:
    votes: list[dict[str, object]] = []
    completed = False
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
                    completed = True
                    await cl.Message(content=render_result(votes, event["content"])).send()
                if event["type"] == "arbitration_error":
                    await cl.Message(content=render_failure(event["content"])).send()
    if completed:
        await cl.Message(content="Send any message when you want to predict churn for another customer.").send()


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
