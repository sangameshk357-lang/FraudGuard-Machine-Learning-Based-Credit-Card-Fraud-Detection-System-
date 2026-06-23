# FraudGuard — Credit Card Fraud Detection

Simple, professional fraud detection app using Logistic Regression.
Model is pre-trained and ready — no configuration needed.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (run once)
python train_model.py

# 3. Start the app
python app.py

# 4. Open browser
#    http://localhost:5050
```

## What's Included

| File | Purpose |
|------|---------|
| `train_model.py` | Trains the model and generates both datasets |
| `app.py` | Flask web server |
| `templates/index.html` | Dashboard UI |
| `data/transactions.csv` | 5,000 transactions (main dataset) |
| `data/test_dataset.csv` | 500 transactions (for verification) |
| `models/` | Saved model, scaler, and metadata |

## Features

- **Main Dataset Tab** — scan all 5,000 transactions, filter by result/risk
- **Test & Verify Tab** — compare predictions vs actual labels (proves model works)
- **Manual Check** — enter Amount + Hour to instantly classify any transaction
- Search, filter by fraud/legit/risk level
- Model stats displayed in the header (AUC, F1, threshold)
