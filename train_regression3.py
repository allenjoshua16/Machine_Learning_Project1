#!/usr/bin/env python3
"""
Train Regression (variant 3)
- Uses sepal_length and petal_width to predict sepal_width
- Performs 10% stratified test split on Iris species
- Trains Linear Regression with batch gradient descent + early stopping
- Saves model weights, diagnostics, and the held-out test split
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from LinearRegression import LinearRegression
from utils import r2_score


def main() -> None:
    data_path = Path(r"C:/Users/allen/Downloads/archive (1)/IRIS.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Expected dataset at {data_path}.")

    features = ["sepal_length", "petal_width"]
    target = "sepal_width"

    df = pd.read_csv(data_path)
    missing = [col for col in features + [target, "species"] if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    species = df["species"].astype(str).str.strip().str.lower()
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        stratify=species,
        random_state=42,
    )

    model = LinearRegression(
        fit_intercept=True,
        lr=0.05,
        degree=1,
        validation_split=0.1,
        random_state=42,
    )
    model.fit(
        X_train,
        y_train,
        batch_size=32,
        regularization=0.0,
        max_epochs=100,
        patience=3,
        shuffle=True,
    )

    test_mse = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print(f"Test MSE : {test_mse:.4f}")
    print(f"Test R^2 : {r2_score(y_test, y_pred):.4f}")

    model.save("model_reg3.npz")
    print("Saved model to model_reg3.npz")

    history = model.history_
    if history is None:
        raise RuntimeError("Training history missing; fit() must record losses.")

    steps = np.arange(1, len(history.train_loss) + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(steps, history.train_loss, label="train", linewidth=2)
    plt.plot(steps, history.val_loss, label="validation", linewidth=2)
    plt.xlabel("Training step")
    plt.ylabel("MSE loss")
    plt.title("Learning Curve - Regression 3")
    plt.legend()
    plt.tight_layout()
    plt.savefig("regression3_learning_curve_mse.png", dpi=300)
    plt.close()
    print("Saved plot to regression3_learning_curve_mse.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.8)
    diag_min = float(min(y_test.min(), y_pred.min()))
    diag_max = float(max(y_test.max(), y_pred.max()))
    plt.plot([diag_min, diag_max], [diag_min, diag_max], linestyle="--", color="red")
    plt.xlabel("True sepal_width")
    plt.ylabel("Predicted sepal_width")
    plt.title("Predicted vs. True - Regression 3")
    plt.tight_layout()
    plt.savefig("regression3_pred_vs_true.png", dpi=300)
    plt.close()
    print("Saved plot to regression3_pred_vs_true.png")

    np.savez(
        "regression3_test_split.npz",
        X_test=X_test,
        y_test=y_test,
        features=np.array(features, dtype=object),
        target=np.array(target, dtype=object),
    )
    print("Saved held-out test split to regression3_test_split.npz")


if __name__ == "__main__":
    main()
