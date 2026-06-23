"""
app.py  —  FraudGuard Flask backend
Run:  python app.py
Open: http://localhost:5050
"""

import os, json
import pandas as pd
import numpy as np
import joblib
from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)
BASE = os.path.dirname(__file__)

# ── Load pre-trained model (once at startup) ───────────────
model  = joblib.load(os.path.join(BASE, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
meta   = json.load(open(os.path.join(BASE, "models", "meta.json")))

FEATURES  = meta["feature_cols"]
THRESHOLD = meta["threshold"]

# ── Helper ─────────────────────────────────────────────────
def classify_row(row: pd.Series) -> dict:
    import warnings; warnings.filterwarnings("ignore")
    x = scaler.transform(pd.DataFrame([row[FEATURES]]))
    prob = float(model.predict_proba(x)[0][1])
    is_fraud = prob >= THRESHOLD
    risk = "HIGH" if prob >= 0.7 else ("MEDIUM" if prob >= 0.4 else "LOW")
    return {
        "probability": round(prob * 100, 1),
        "is_fraud":    is_fraud,
        "risk":        risk,
        "label":       "FRAUD" if is_fraud else "LEGITIMATE",
    }

def load_csv(name: str) -> pd.DataFrame:
    path = os.path.join(BASE, "data", name)
    return pd.read_csv(path)

def df_to_records(df: pd.DataFrame) -> list:
    """Return list of dicts with classification for each row."""
    out = []
    for _, row in df.iterrows():
        c = classify_row(row)
        out.append({
            "id":       row.get("TransactionID", "—"),
            "time":     row.get("Timestamp",     "—"),
            "card":     "•••• " + str(row.get("CardLast4", "????"))[-4:],
            "merchant": row.get("MerchantCategory", "—"),
            "amount":   round(float(row.get("Amount", 0)), 2),
            "hour":     int(row.get("Hour", 0)),
            "label":    c["label"],
            "risk":     c["risk"],
            "prob":     c["probability"],
            "actual":   int(row.get("Class", -1)),   # -1 = unknown
        })
    return out

# ── Routes ─────────────────────────────────────────────────
HTML = open(os.path.join(BASE, "templates", "index.html"), encoding="utf-8").read()

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/api/summary")
def summary():
    """Return model stats and dataset summary."""
    main  = load_csv("transactions.csv")
    test  = load_csv("test_dataset.csv")
    return jsonify({
        "model": {
            "roc_auc":   meta["roc_auc"],
            "f1":        meta["f1"],
            "threshold": round(THRESHOLD * 100, 1),
            "algorithm": "Logistic Regression",
            "status":    "Trained & Ready",
        },
        "main_dataset":  {"rows": len(main),  "fraud": int(main["Class"].sum()),  "name": "transactions.csv"},
        "test_dataset":  {"rows": len(test),   "fraud": int(test["Class"].sum()),  "name": "test_dataset.csv"},
    })

@app.route("/api/transactions")
def transactions():
    """Scan the main dataset and return all classified transactions."""
    source = request.args.get("source", "main")
    fname  = "test_dataset.csv" if source == "test" else "transactions.csv"
    df     = load_csv(fname)
    records = df_to_records(df)
    fraud_count = sum(1 for r in records if r["label"] == "FRAUD")
    return jsonify({
        "records":     records,
        "total":       len(records),
        "fraud_count": fraud_count,
        "legit_count": len(records) - fraud_count,
        "source":      fname,
    })

@app.route("/api/scan_single", methods=["POST"])
def scan_single():
    """Classify a single manually-entered transaction."""
    data = request.json or {}
    try:
        row_data = {f: float(data.get(f, 0)) for f in FEATURES}
        row = pd.Series(row_data)
        result = classify_row(row)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    print("\n  FraudGuard is running →  http://localhost:5050\n")
    app.run(debug=False, port=5050)
