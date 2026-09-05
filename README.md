# Ensemble Verdict Churn

This academic, non-commercial MVP makes a local churn ensemble observable. It runs five persisted classifiers and calls Qwen only when their votes split 3–2.

## Run the demonstration

The versioned dataset is included locally, so training does not require Kaggle credentials or network access.

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[training,test]"
python training/train.py
docker compose up --build
```

The training stack uses Python 3.13 and PyCaret 4.0.0a8, the newest PyCaret release compatible with the modern scientific Python stack. Python 3.14 is not supported yet by its joblib and cloudpickle persistence dependencies.

Open `http://localhost:8001` and submit the generated customer form. Its fields and permitted values are derived from the persisted training schema.

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

`training/train.py` reads the versioned local file `data/ecommerce_customer_churn.csv`. It removes `CustomerID`, normalizes categorical values, applies train-only imputation and SMOTE, freezes a stratified 80/20 split using seed 42, and applies PyCaret feature selection. It persists KNN, SVM-RBF, Random Forest, XGBoost, and Naive Bayes pipelines plus prepared data, splits, schema, predictions, and model/ensemble metrics.

The local dataset is derived from Kaggle dataset `ankitverma2010/ecommerce-customer-churn-analysis-and-prediction`, version 1, file `E Commerce Dataset.xlsx`, sheet `E Comm`. It is licensed CC BY-NC-SA 4.0 and this repository is for study and non-commercial demonstration only. See `data/DATASET_LICENSE.md` for attribution. Dataset quality, historical bias, and the small local arbiter limit the reliability of any real-world use.

## Tests

```powershell
python -m pytest
```
