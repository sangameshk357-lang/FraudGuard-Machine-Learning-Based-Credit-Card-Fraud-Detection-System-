"""
train_model.py
==============
Run this ONCE to train the Logistic Regression model and generate
two ready-to-use CSV datasets:
  data/transactions.csv  — 5,000 transactions (used by the main app)
  data/test_dataset.csv  — 500 transactions (used to verify the app works)

Usage:
    python train_model.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib, os, json

SEED = 42
np.random.seed(SEED)

print("=" * 55)
print("  FraudGuard — Model Training")
print("=" * 55)

# ── 1. Generate realistic synthetic data ───────────────────
def make_data(n=6000, fraud_rate=0.017, seed=42):
    rng = np.random.RandomState(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    def legit_row():
        v = rng.randn(28)
        amount = abs(rng.exponential(60))
        hour   = rng.choice(range(8, 22))          # daytime
        return list(v) + [round(amount, 2), hour]

    def fraud_row():
        v = rng.randn(28) * 1.6 + rng.choice([-2, 2], 28) * rng.randint(0, 2, 28)
        amount = abs(rng.exponential(250))          # larger amounts
        hour   = rng.choice(list(range(0, 6)) + list(range(22, 24)))  # off-hours
        return list(v) + [round(amount, 2), hour]

    cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]

    rows = [legit_row() for _ in range(n_legit)] + [fraud_row() for _ in range(n_fraud)]
    labels = [0] * n_legit + [1] * n_fraud

    df = pd.DataFrame(rows, columns=cols)
    df["Class"] = labels

    # Shuffle
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Add readable metadata
    base_ts = pd.Timestamp("2024-01-01 00:00:00")
    df["TransactionID"] = [f"TXN{str(i).zfill(6)}" for i in range(len(df))]
    df["Timestamp"] = [
        (base_ts + pd.Timedelta(hours=int(h), minutes=int(rng.randint(0, 60)))).strftime("%Y-%m-%d %H:%M")
        for h in df["Hour"]
    ]
    df["CardLast4"] = [str(rng.randint(1000, 9999)) for _ in range(len(df))]
    df["MerchantCategory"] = rng.choice(
        ["Grocery", "Electronics", "Restaurant", "Travel", "Online Shopping",
         "Gas Station", "Pharmacy", "ATM Withdrawal", "Hotel", "Luxury Goods"],
        size=len(df)
    )
    return df

print("\n[1/4] Generating datasets...")
full_df = make_data(n=6000, fraud_rate=0.017, seed=SEED)

# Main dataset: 5000 rows
main_df = full_df.iloc[:5000].copy()
main_df.to_csv("data/transactions.csv", index=False)
print(f"      transactions.csv  → {len(main_df):,} rows  | fraud: {main_df['Class'].sum()} ({main_df['Class'].mean()*100:.1f}%)")

# Test/verification dataset: 500 rows, slightly higher fraud rate so it's useful
test_df = make_data(n=500, fraud_rate=0.08, seed=SEED + 1)
test_df.to_csv("data/test_dataset.csv", index=False)
print(f"      test_dataset.csv  → {len(test_df):,} rows  | fraud: {test_df['Class'].sum()} ({test_df['Class'].mean()*100:.1f}%)")

# ── 2. Prepare training data ───────────────────────────────
print("\n[2/4] Training Logistic Regression model...")
feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Hour"]

X = main_df[feature_cols]
y = main_df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# SMOTE to handle imbalance
smote = SMOTE(random_state=SEED, k_neighbors=5)
X_res, y_res = smote.fit_resample(X_train_sc, y_train)

model = LogisticRegression(C=0.5, solver="lbfgs", max_iter=2000,
                           class_weight="balanced", random_state=SEED)
model.fit(X_res, y_res)

# ── 3. Find best threshold ─────────────────────────────────
y_prob = model.predict_proba(X_test_sc)[:, 1]
from sklearn.metrics import f1_score
best_t, best_f1 = 0.5, 0
for t in np.linspace(0.05, 0.95, 200):
    f = f1_score(y_test, (y_prob >= t).astype(int), zero_division=0)
    if f > best_f1:
        best_f1, best_t = f, t

y_pred = (y_prob >= best_t).astype(int)
auc    = roc_auc_score(y_test, y_prob)
cm     = confusion_matrix(y_test, y_pred)

print(f"\n      ROC-AUC  : {auc:.4f}")
print(f"      F1 Score : {best_f1:.4f}")
print(f"      Threshold: {best_t:.3f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legitimate','Fraud'])}")

# ── 4. Save artefacts ──────────────────────────────────────
print("[3/4] Saving model artefacts...")
joblib.dump(model,  "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

meta = {
    "feature_cols": feature_cols,
    "threshold":    round(float(best_t), 4),
    "roc_auc":      round(float(auc), 4),
    "f1":           round(float(best_f1), 4),
    "cm": {"tn": int(cm[0,0]), "fp": int(cm[0,1]),
           "fn": int(cm[1,0]), "tp": int(cm[1,1])},
    "coefs": dict(zip(feature_cols, model.coef_[0].tolist()))
}
json.dump(meta, open("models/meta.json", "w"), indent=2)

print("      models/model.pkl  ✓")
print("      models/scaler.pkl ✓")
print("      models/meta.json  ✓")

print("\n[4/4] Done! Run  python app.py  to start the app.")
print("=" * 55)
