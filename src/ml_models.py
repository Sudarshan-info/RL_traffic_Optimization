import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

FEATURES = ["arrival_rate", "queue_length", "hour", "is_weekend"]
TARGET = "green_time"


def train_all_models(data_path="data/traffic_data.csv"):
    df = pd.read_csv(data_path)
    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models = {
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "LinearRegression": LinearRegression(),
        "SVR": SVR(kernel="rbf", C=10, epsilon=0.5),
    }
    results = {}
    Path("models").mkdir(exist_ok=True)

    for name, model in models.items():
        print(f"Training {name}...")
        scaled = name in ["LinearRegression", "SVR"]
        model.fit(X_tr_s if scaled else X_tr, y_tr)
        preds = model.predict(X_te_s if scaled else X_te)
        mae = mean_absolute_error(y_te, preds)
        r2 = r2_score(y_te, preds)
        results[name] = {"mae": round(mae, 3), "r2": round(r2, 3)}
        print(f"  {name}: MAE={mae:.3f}  R2={r2:.3f}")
        joblib.dump(model, f"models/{name.lower()}.pkl")

    return results
