#!/usr/bin/env python3
"""
Evaluate Regression (variant 2)
- Loads trained Linear Regression (model_reg2.npz)
- Hard-coded CSV path and columns
- Prints RMSE and R^2 on the full dataset
"""

import pandas as pd
import numpy as np
from utils import rmse, r2_score
from LinearRegression import LinearRegression

def main():
    # === Must match train_regression2.py ===
    data_path = r"C:/Users/allen/Downloads/archive (1)/IRIS.csv"
    model_path = "model_reg2.npz"
    features = ["petal_length"]
    target = "sepal_length"

    # Load data
    df = pd.read_csv(data_path)
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)

    # Load model & predict
    model = LinearRegression.load(model_path)
    yhat = model.predict(X)

    # Metrics
    print(f"RMSE: {rmse(y, yhat):.4f}")
    print(f"R^2 : {r2_score(y, yhat):.4f}")

if __name__ == "__main__":
    main()
