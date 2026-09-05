# Ensemble Verdict Churn

This academic, non-commercial MVP makes a local churn ensemble observable. It runs five persisted classifiers and calls Qwen only when their votes split 3–2.

## Run the demonstration

Train the artifacts first. Kaggle credentials must be configured locally and are never committed.

```powershell
python -m pip install -e ".[training,test]"
python training/train.py
docker compose up --build
```

Python 3.14 is not supported by the pinned PyCaret/NumPy training stack. The Docker images already use Python 3.12; for local training, create a Python 3.12 virtual environment first.

Open `http://localhost:8001`. Send a JSON customer profile using the field names and values from `http://localhost:8000/form-schema`.

Pull Qwen once after Compose starts:

```powershell
docker compose exec ollama ollama pull qwen3:1.7b
```

For NVIDIA GPU acceleration, use `docker compose -f docker-compose.yml -f compose.gpu.yml up --build`.

## Decision policy

| Vote split | Outcome |
| --- | --- |
| 5–0 | Ensemble unanimity |
| 4–1 | Ensemble consensus |
| 3–2 | Qwen arbitration |

An unavailable or invalid arbitration emits an explicit SSE error and no final churn label.

## Training and reproducibility

`training/train.py` downloads the immutable Kaggle handle `ankitverma2010/ecommerce-customer-churn-analysis-and-prediction/versions/1`, source file `E Commerce Dataset.xlsx`, sheet `E Comm`. It removes `CustomerID`, normalizes categorical values, applies mean/mode imputation, freezes a stratified 80/20 split using seed 42, and uses PyCaret feature selection and SMOTE within training only. It persists KNN, SVM-RBF, Random Forest, XGBoost, and Naive Bayes pipelines plus prepared data, splits, schema, predictions, and model/ensemble metrics.

The dataset is attributed to its Kaggle publisher and is licensed CC BY-NC-SA 4.0. This repository is for study and non-commercial demonstration only. Dataset quality, historical bias, and the small local arbiter limit the reliability of any real-world use.

## Tests

```powershell
python -m pytest
```
