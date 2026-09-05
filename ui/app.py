import json
import os

import chainlit as cl
import httpx
from chainlit.input_widget import Select, TextInput

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
        await cl.ChatSettings(create_form(schema)).send()
        await cl.Message(
            content="Complete the customer form and save settings to run a prediction."
        ).send()
    except httpx.HTTPError as error:
        await cl.Message(content=f"The prediction form is unavailable: {error}").send()


@cl.on_message
async def predict(message: cl.Message) -> None:
    try:
        instance = json.loads(message.content)
    except json.JSONDecodeError:
        await cl.Message(content="Send a valid JSON customer profile.").send()
        return
    await stream_prediction(instance)


@cl.on_settings_update
async def predict_from_form(settings: dict[str, object]) -> None:
    await stream_prediction(coerce_numbers(settings))


def create_form(schema: dict[str, object]) -> list[Select | TextInput]:
    fields = schema.get("fields", [])
    widgets: list[Select | TextInput] = []
    for field in fields:
        if field["type"] == "select":
            widgets.append(Select(id=field["name"], label=field["name"], values=field["options"]))
        else:
            widgets.append(
                TextInput(id=field["name"], label=field["name"], initial=str(field["minimum"]))
            )
    return widgets


def coerce_numbers(settings: dict[str, object]) -> dict[str, object]:
    return {
        key: float(value)
        if isinstance(value, str) and value.replace(".", "", 1).isdigit()
        else value
        for key, value in settings.items()
    }


async def stream_prediction(instance: dict[str, object]) -> None:
    votes: list[dict[str, object]] = []
    async with httpx.AsyncClient(timeout=90) as client:
        async with client.stream(
            "POST", f"{API_URL}/predictions/stream", json=instance
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    event = json.loads(line[6:])
                    if event["type"] == "model_vote":
                        votes.append(event["content"])
                    async with cl.Step(name=event["type"], type="tool") as step:
                        step.output = format_event(event)
                    if event["type"] == "decision":
                        await cl.Message(content=render_result(votes, event["content"])).send()


def format_event(event: dict[str, object]) -> str:
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
