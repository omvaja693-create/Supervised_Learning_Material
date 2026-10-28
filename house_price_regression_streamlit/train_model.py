"""
Train and export the complete preprocessing + regression pipeline.

Run:
    python train_model.py

Output:
    model/house_price_pipeline.pkl
    model/metrics.json
"""
import json
import pickle
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


FEATURES = ["area", "bedrooms", "bathrooms", "age"]
TARGET = "price"


def main():
    df = pd.read_csv("data/house_data.csv")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("numeric", numeric_pipeline, FEATURES)]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LinearRegression()),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "MSE": float(mse),
        "RMSE": float(mse ** 0.5),
        "R2": float(r2_score(y_test, y_pred)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "target": TARGET,
    }

    with open("model/house_price_pipeline.pkl", "wb") as file:
        pickle.dump(pipeline, file)

    with open("model/metrics.json", "w") as file:
        json.dumps(metrics, indent=4)

    # "model/metrics.json".write_text(json.dumps(metrics, indent=4))

    print("\nModel trained successfully.")
    print("Pipeline: model/house_price_pipeline.pkl")
    print("\nEvaluation:")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
