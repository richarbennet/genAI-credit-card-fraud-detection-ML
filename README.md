
# 🛡️ Real-Time Credit Card Fraud Detection with GenAI + ML

## 🚀 Overview

This project builds a **production-ready system** for detecting credit card fraud in real-time. It combines anomaly detection with explainable AI (XAI) and real-time API deployment. Key technologies include:

- 🧠 **Isolation Forest, Logistic Regression** (switchable via `config.yaml`)
- 🔍 **Explainability with SHAP**
- 🧩 Modular ML pipelines (preprocessing, feature engineering, training, API)
- 🌐 **Flask-based REST API** for real-time inference

## 📦 Dataset

- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Highly imbalanced: 492 frauds out of 284,807 transactions
- Features: PCA-transformed `V1–V28`, plus `Time`, `Amount`, `Class`

👉 **Put `creditcard.csv` in** `data/raw/`

## 🔍 Methodology

### 1. Data Ingestion
- Clean and split data
- Extract meaningful time-based features

### 2. Feature Engineering
- Adds custom features like `Amount_relative_to_mean`, `Hour`, etc.

### 3. Model Training
- Default: `IsolationForest`
- Metrics: Precision, Recall, F1, Average Precision
- Models saved under `models/`

### 4. Real-Time Inference
- `api.py` hosts a REST API via Flask
- Uses SHAP for on-the-fly explanation of results

### 5. Testing
- Located in `tests/`
- Run with: `python -m unittest discover tests`

## ⚙️ Configuration

All behavior is driven by `config.yaml`:

```yaml
train:
  model: "IsolationForest"
  n_estimators: 150
  contamination: 0.01
```

## ▶️ Quickstart

```bash
git clone <repo_url>
cd real-time-credit-card-fraud-detection-with-genAI-ML-main
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Put `creditcard.csv` in `data/raw/`

Train model:
```bash
python src/training.py
```

Run API:
```bash
python src/api.py
```

Test:
```bash
python -m unittest discover tests
```

## 🧪 Notebooks

Explore and validate in:
- `notebooks/data_exploration.ipynb`
- `notebooks/model_development.ipynb`
- `notebooks/explainability.ipynb`

## 📁 Repo Structure (Proposed Clean Format)

```
real-time-fraud-detection/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── src/
│   ├── api.py
│   ├── training.py
│   ├── feature_engineering.py
│   └── inference.py
├── tests/
├── config.yaml
├── requirements.txt
└── README.md
```

## 🧠 Highlights

- GenAI-ready structure
- Real-time predictions with low latency
- SHAP-based interpretability
- Modular & testable ML system

## 🙌 Credits

Developed by **Richard R. Bennet**  
📧 ri.be.1872@gmail.com | 🌐 [LinkedIn](https://www.linkedin.com/in/richard-r-b-542831351/)
