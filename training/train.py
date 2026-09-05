import json
import time
from pathlib import Path

import kagglehub
import pandas as pd
from pycaret.classification import create_model, predict_model, save_model, setup
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.services import artifact_filename

DATASET_HANDLE = "ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/versions/1"
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


def evaluate_predictions(
    test: pd.DataFrame, predictions: pd.DataFrame, latency_ms: float
) -> dict[str, object]:
    actual = test["Churn"]
    predicted = predictions["prediction_label"]
    score = predictions["prediction_score"]
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
    dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    source = next(dataset_path.rglob("E Commerce Dataset.xlsx"))
    data = normalize_data(pd.read_excel(source, sheet_name="E Comm"))
    data.to_parquet(output / "prepared_data.parquet", index=False)
    schema_path = output / "form_schema.json"
    schema_path.write_text(json.dumps(create_form_schema(data), indent=2), encoding="utf-8")
    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Churn"])
    train.to_parquet(output / "train_split.parquet", index=False)
    test.to_parquet(output / "test_split.parquet", index=False)
    setup(
        data=train,
        target="Churn",
        session_id=42,
        fix_imbalance=True,
        fix_imbalance_method="SMOTE",
        numeric_imputation="mean",
        categorical_imputation="mode",
        feature_selection=True,
        feature_selection_method="classic",
        fold_strategy="stratifiedkfold",
        verbose=False,
    )
    metrics: dict[str, object] = {}
    predictions_by_model: dict[str, pd.DataFrame] = {}
    for name, model_id in MODEL_IDS.items():
        model = create_model(model_id, verbose=False)
        save_model(model, str(output / artifact_filename(name)))
        started_at = time.perf_counter()
        predictions = predict_model(model, data=test, verbose=False)
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
