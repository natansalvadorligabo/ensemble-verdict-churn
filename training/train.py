import json
import time
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from pycaret.classification import ClassificationExperiment
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from app.services import artifact_filename

DATASET_PATH = Path("data/ecommerce_customer_churn.csv")
MODEL_IDS = {
    "KNN": "knn",
    "SVM-RBF": "rbfsvm",
    "Random Forest": "rf",
    "XGBoost": "xgboost",
    "Naive Bayes": "nb",
}


def normalize_data(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned.columns = cleaned.columns.str.strip()
    for column in cleaned.select_dtypes(include="object"):
        normalized = cleaned[column].astype("string").str.strip()
        cleaned[column] = normalized.replace({"Mobile Phone": "Mobile"})
    return cleaned.drop(columns=["CustomerID"], errors="ignore")


def create_form_schema(frame: pd.DataFrame) -> dict[str, object]:
    fields = []
    for name in frame.columns:
        if name == "Churn":
            continue
        series = frame[name]
        field: dict[str, object] = {"name": name}
        if pd.api.types.is_numeric_dtype(series):
            field |= {
                "type": "number",
                "minimum": float(series.min()),
                "maximum": float(series.max()),
            }
        else:
            field |= {
                "type": "select",
                "options": sorted(series.dropna().astype(str).unique().tolist()),
            }
        fields.append(field)
    return {"fields": fields}


def resample_training_data(train: pd.DataFrame) -> pd.DataFrame:
    features = train.drop(columns=["Churn"])
    target = train["Churn"]
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_columns = features.columns.difference(categorical_columns).tolist()
    encoded = features.copy()
    if numeric_columns:
        numeric_imputer = SimpleImputer(strategy="median")
        encoded[numeric_columns] = numeric_imputer.fit_transform(features[numeric_columns])
    encoder: OrdinalEncoder | None = None
    if categorical_columns:
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        imputed_categories = categorical_imputer.fit_transform(features[categorical_columns])
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        encoded[categorical_columns] = encoder.fit_transform(imputed_categories)
    minority_count = int(target.value_counts().min())
    if minority_count < 2:
        raise ValueError("SMOTE requires at least two minority-class training samples")
    neighbors = min(5, minority_count - 1)
    if categorical_columns:
        categorical_indexes = [encoded.columns.get_loc(column) for column in categorical_columns]
        sampler = SMOTENC(
            categorical_features=categorical_indexes,
            k_neighbors=neighbors,
            random_state=42,
        )
    else:
        sampler = SMOTE(k_neighbors=neighbors, random_state=42)
    resampled_features, resampled_target = sampler.fit_resample(encoded, target)
    resampled = pd.DataFrame(resampled_features, columns=features.columns)
    if encoder is not None:
        resampled[categorical_columns] = encoder.inverse_transform(resampled[categorical_columns])
    return pd.concat([resampled, pd.Series(resampled_target, name="Churn")], axis=1)


def evaluate_predictions(
    test: pd.DataFrame, predictions: pd.DataFrame, latency_ms: float
) -> dict[str, object]:
    actual = test["Churn"]
    predicted = predictions["prediction_label"]
    score_columns = sorted(
        column for column in predictions.columns if column.startswith("prediction_score_")
    )
    score = predictions[score_columns[-1]] if score_columns else predictions["prediction_score"]
    return {
        "precision": precision_score(actual, predicted, average="weighted", zero_division=0),
        "recall": recall_score(actual, predicted, average="weighted", zero_division=0),
        "f1": f1_score(actual, predicted, average="weighted", zero_division=0),
        "roc_auc": roc_auc_score(actual, score),
        "confusion_matrix": confusion_matrix(actual, predicted).tolist(),
        "average_latency_ms": latency_ms / len(test),
    }


def ensemble_rates(predictions_by_model: dict[str, pd.DataFrame]) -> dict[str, float]:
    labels = pd.DataFrame(
        {
            name: predictions["prediction_label"].reset_index(drop=True)
            for name, predictions in predictions_by_model.items()
        }
    )
    maximum_votes = labels.apply(lambda row: row.value_counts().iloc[0], axis=1)
    total = len(labels)
    return {
        "unanimity": float((maximum_votes == 5).sum() / total),
        "consensus": float((maximum_votes == 4).sum() / total),
        "arbitration": float((maximum_votes == 3).sum() / total),
    }


def main() -> None:
    output = Path("artifacts")
    output.mkdir(exist_ok=True)
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset is missing: {DATASET_PATH}")
    data = normalize_data(pd.read_csv(DATASET_PATH))
    data.to_parquet(output / "prepared_data.parquet", index=False)
    schema_path = output / "form_schema.json"
    schema_path.write_text(json.dumps(create_form_schema(data), indent=2), encoding="utf-8")
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Churn"])
    train.to_parquet(output / "train_split.parquet", index=False)
    test.to_parquet(output / "test_split.parquet", index=False)
    resampled_train = resample_training_data(train)
    experiment = ClassificationExperiment(
        target="Churn",
        session_id=42,
        fold_strategy="stratifiedkfold",
        feature_selection=True,
        n_jobs=1,
        verbose=False,
    ).fit(resampled_train.drop(columns=["Churn"]), resampled_train["Churn"])
    metrics: dict[str, object] = {}
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for name, model_id in MODEL_IDS.items():
        model = experiment.create_model(model_id).pipeline
        experiment.save_model(model, output / artifact_filename(name))
        started_at = time.perf_counter()
        predictions = experiment.predict_model(
            model, test.drop(columns=["Churn"]), raw_score=True
        ).predictions
        latency_ms = (time.perf_counter() - started_at) * 1000
        metrics[name] = evaluate_predictions(test, predictions, latency_ms)
        predictions_by_model[name] = predictions
        prediction_path = output / f"{artifact_filename(name)}_predictions.parquet"
        predictions.to_parquet(prediction_path)
    metrics_path = output / "model_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    ensemble_path = output / "ensemble_metrics.json"
    ensemble_path.write_text(
        json.dumps(ensemble_rates(predictions_by_model), indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
