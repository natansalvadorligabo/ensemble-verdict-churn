import pytest

from app.domain import ModelVote, aggregate_votes
from app.services import normalize_arbitration_response, validate_instance


def votes(labels: list[str]) -> list[ModelVote]:
    return [ModelVote(f"Model {index}", label, 0.8, 2.0) for index, label in enumerate(labels)]


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        (["No", "No", "No", "No", "No"], "unanimity"),
        (["No", "No", "No", "No", "Yes"], "consensus"),
        (["No", "No", "No", "Yes", "Yes"], "arbitration"),
    ],
)
def test_aggregate_votes_classifies_the_agreement(labels: list[str], expected: str) -> None:
    assert aggregate_votes(votes(labels)).agreement == expected


def test_aggregate_votes_requires_five_models() -> None:
    with pytest.raises(ValueError, match="Exactly five"):
        aggregate_votes(votes(["No"]))


def test_arbitration_response_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="outside"):
        normalize_arbitration_response('{"label":"Maybe","explanation":"Unknown"}', {"No", "Yes"})


def test_arbitration_response_requires_explanation() -> None:
    with pytest.raises(ValueError, match="explanation"):
        normalize_arbitration_response('{"label":"No","explanation":""}', {"No", "Yes"})


def test_instance_validation_accepts_only_the_trained_schema() -> None:
    schema: dict[str, object] = {
        "fields": [
            {"name": "Tenure", "type": "number", "minimum": 0, "maximum": 10},
            {"name": "Device", "type": "select", "options": ["Mobile", "Desktop"]},
        ]
    }

    assert validate_instance({"Tenure": "4", "Device": "Mobile"}, schema) == {
        "Tenure": 4.0,
        "Device": "Mobile",
    }


def test_instance_validation_rejects_untrained_categories() -> None:
    schema: dict[str, object] = {
        "fields": [{"name": "Device", "type": "select", "options": ["Mobile"]}]
    }

    with pytest.raises(ValueError, match="trained category"):
        validate_instance({"Device": "Tablet"}, schema)
