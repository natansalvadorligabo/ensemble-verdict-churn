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


async def extract_profile(
    message: str, current_profile: dict[str, object]
) -> dict[str, object]:
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"{API_URL}/profiles/extract",
            json={"message": message, "current_profile": current_profile},
        )
        response.raise_for_status()
        return response.json()


async def interpret_message(message: str, context: dict[str, object]) -> str:
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"{API_URL}/conversation/interpret",
            json={"message": message, "context": context},
        )
        response.raise_for_status()
        return str(response.json()["intent"])


@cl.on_chat_start
async def start() -> None:
    try:
        cl.user_session.set("schema", await request_schema())
        await cl.Message(
            content=(
                "Describe the customer naturally, in any language. I will extract the model "
                "inputs, ask only for missing information, and show the profile for confirmation. "
                "Type `/form` at any time to use the manual form instead."
            )
        ).send()
    except httpx.HTTPError as error:
        await cl.Message(content=f"The prediction assistant is unavailable: {error}").send()


@cl.on_message
async def handle_message(message: cl.Message) -> None:
    content = message.content.strip()
    schema = await ensure_schema()
    if schema is None:
        return
    if content.casefold() == "/form":
        cl.user_session.set("last_analysis", None)
        await handle_form(schema)
        return
    if content.casefold() == "/new":
        reset_profile()
        cl.user_session.set("last_analysis", None)
        await cl.Message(content="Describe the new customer when ready.").send()
        return
    pending_profile = cl.user_session.get("pending_profile")
    draft_profile = cl.user_session.get("draft_profile")
    last_analysis = cl.user_session.get("last_analysis")
    try:
        async with cl.Step(name="interpret_message", type="tool") as step:
            intent = await interpret_message(
                content,
                {
                    "profile_awaiting_confirmation": isinstance(pending_profile, dict),
                    "profile_in_progress": isinstance(draft_profile, dict),
                    "completed_prediction_available": isinstance(last_analysis, dict),
                },
            )
            step.output = f"Intent: {intent}"
    except httpx.HTTPError as error:
        await cl.Message(content=f"I could not interpret the message: {error}").send()
        return
    if isinstance(pending_profile, dict):
        if intent == "confirm_profile":
            cl.user_session.set("pending_profile", None)
            await stream_prediction(pending_profile)
            await invite_another_prediction()
            return
        if intent == "cancel_profile":
            reset_profile()
            await cl.Message(content="Prediction cancelled. Describe another customer when ready.").send()
            return
        await process_description(content, pending_profile, schema)
        return
    if isinstance(draft_profile, dict):
        if intent == "cancel_profile":
            reset_profile()
            await cl.Message(content="Prediction cancelled. Describe another customer when ready.").send()
            return
        await process_description(content, draft_profile, schema)
        return
    if isinstance(last_analysis, dict) and intent == "ask_about_result":
        await answer_follow_up(content, last_analysis)
        return
    cl.user_session.set("last_analysis", None)
    await process_description(content, {}, schema)


async def ensure_schema() -> dict[str, object] | None:
    schema = cl.user_session.get("schema")
    if isinstance(schema, dict):
        return schema
    try:
        schema = await request_schema()
        cl.user_session.set("schema", schema)
        return schema
    except httpx.HTTPError as error:
        await cl.Message(content=f"The prediction assistant is unavailable: {error}").send()
        return None


async def process_description(
    content: str,
    current_profile: dict[str, object],
    schema: dict[str, object],
) -> None:
    try:
        async with cl.Step(name="extract_customer_profile", type="tool") as step:
            extraction = await extract_profile(content, current_profile)
            step.output = "Customer attributes extracted and validated."
    except httpx.HTTPError as error:
        await cl.Message(content=f"I could not interpret the customer description: {error}").send()
        return
    profile = extraction.get("profile", {})
    missing = extraction.get("missing_fields", [])
    if not isinstance(profile, dict) or not isinstance(missing, list):
        await cl.Message(content="The extracted customer profile is invalid.").send()
        return
    if missing:
        cl.user_session.set("draft_profile", profile)
        await cl.Message(content=render_missing_fields(missing)).send()
        return
    cl.user_session.set("draft_profile", None)
    cl.user_session.set("pending_profile", profile)
    await cl.Message(content=render_profile_confirmation(profile, schema)).send()


async def answer_follow_up(question: str, analysis: dict[str, object]) -> None:
    try:
        message = cl.Message(content="")
        async with cl.Step(name="explain_prediction", type="tool") as step:
            async with httpx.AsyncClient(timeout=90) as client:
                async with client.stream(
                    "POST",
                    f"{API_URL}/predictions/explain/stream",
                    json={"question": question, "analysis": analysis},
                ) as response:
                    response.raise_for_status()
                    async for token in response.aiter_text():
                        await message.stream_token(token)
            step.output = "The previous prediction was reviewed."
        await message.send()
    except httpx.HTTPError as error:
        await cl.Message(content=f"I could not explain the previous prediction: {error}").send()


async def handle_form(schema: dict[str, object]) -> None:
    reset_profile()
    instance = await request_customer_profile(schema)
    if instance is None:
        await cl.Message(content="The prediction was cancelled.").send()
        return
    await stream_prediction(instance)
    await invite_another_prediction()


def reset_profile() -> None:
    cl.user_session.set("draft_profile", None)
    cl.user_session.set("pending_profile", None)


def render_missing_fields(missing: list[object]) -> str:
    names = ", ".join(f"**{humanize(str(name))}**" for name in missing)
    return (
        "I captured the information provided. Please send the remaining details in one message: "
        f"{names}. You can also type `/form`."
    )


def render_profile_confirmation(
    profile: dict[str, object], schema: dict[str, object]
) -> str:
    fields = schema.get("fields", [])
    order = [str(field["name"]) for field in fields if isinstance(field, dict)]
    rows = "\n".join(f"| {humanize(name)} | {profile[name]} |" for name in order)
    return (
        "## Confirm customer profile\n\n"
        "| Attribute | Value |\n"
        "| --- | --- |\n"
        f"{rows}\n\n"
        "Reply **confirm** to run the prediction, send corrections naturally, or reply **cancel**."
    )


async def invite_another_prediction() -> None:
    reset_profile()
    await cl.Message(
        content=(
            "Ask me anything about this result. To continue, describe another customer, "
            "type `/new`, or type `/form` to use the manual form."
        )
    ).send()


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
                    cl.user_session.set(
                        "last_analysis",
                        {
                            "profile": instance,
                            "votes": votes,
                            "decision": event["content"],
                        },
                    )
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
