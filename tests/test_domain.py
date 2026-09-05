import pytest

from app.domain import ModelVote, aggregate_votes
from app.services import (
    ArtifactEnsemble,
    arbitration_schema,
    normalize_arbitration_response,
    normalize_profile_extraction,
    profile_extraction_schema,
    validate_instance,
)


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


def test_arbitration_schema_restricts_the_response_to_allowed_labels() -> None:
    schema = arbitration_schema({"1", "0"})

    assert schema["properties"]["label"]["enum"] == ["0", "1"]
    assert schema["required"] == ["label", "explanation"]
    assert schema["additionalProperties"] is False


def test_profile_extraction_schema_allows_explicitly_missing_fields() -> None:
    form_schema = {
        "fields": [
            {"name": "Tenure", "type": "number", "minimum": 0, "maximum": 10},
            {"name": "Device", "type": "select", "options": ["Mobile", "Computer"]},
        ]
    }

    schema = profile_extraction_schema(form_schema)

    profile = schema["properties"]["profile"]
    assert profile["required"] == ["Tenure", "Device"]
    assert profile["properties"]["Tenure"]["anyOf"][1] == {"type": "null"}
    assert profile["properties"]["Device"]["anyOf"][0]["enum"] == ["Mobile", "Computer"]


def test_profile_extraction_computes_missing_fields_from_validated_values() -> None:
    form_schema = {
        "fields": [
            {"name": "Tenure", "type": "number", "minimum": 0, "maximum": 10},
            {"name": "Device", "type": "select", "options": ["Mobile", "Computer"]},
        ]
    }

    extraction = normalize_profile_extraction(
        {"profile": {"Tenure": 4, "Device": None}}, form_schema
    )

    assert extraction == {"profile": {"Tenure": 4.0}, "missing_fields": ["Device"]}


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


@pytest.mark.asyncio
async def test_artifact_ensemble_passes_a_tabular_instance_to_the_pipeline() -> None:
    class TabularModel:
        classes_ = ["No", "Yes"]

        def predict(self, frame: object) -> list[str]:
            assert frame.__class__.__name__ == "DataFrame"
            assert frame.iloc[0]["Tenure"] == 4
            return ["No"]

        def predict_proba(self, frame: object) -> list[list[float]]:
            return [[0.9, 0.1]]

    ensemble = ArtifactEnsemble.__new__(ArtifactEnsemble)
    ensemble.models = {"KNN": TabularModel()}

    votes = [vote async for vote in ensemble.predict({"Tenure": 4})]

    assert votes[0].model == "KNN"
    assert votes[0].label == "No"
    assert votes[0].confidence == 0.9
