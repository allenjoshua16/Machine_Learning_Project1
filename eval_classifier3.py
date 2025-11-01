#!/usr/bin/env python3
"""
Evaluate Classifier (variant 3)
- Loads trained binary logistic regression (model_cls3.npz)
- Hard-coded CSV path and features (all four: sepal_length, sepal_width, petal_length, petal_width)
- Creates species_binary: Iris-versicolor = 1, others = 0
- Prints Accuracy, Precision, Recall, F1
"""

import pandas as pd
import numpy as np
from utils import accuracy_score, precision_recall_f1
from LogisticRegression import LogisticRegression

def main():
    # === Hard-coded CSV and model path ===
    data_path = r"C:/Users/allen/Downloads/archive (1)/IRIS.csv"
    model_path = "model_cls3.npz"
    threshold = 0.5

    # Features used by train_classifier3.py
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # Load data
    df = pd.read_csv(data_path)

    # Create binary target: versicolor = 1, others = 0 (robust to case/whitespace)
    species_lower = df["species"].astype(str).str.strip().str.lower()
    df["species_binary"] = (species_lower.str.contains("versicolor")).astype(int)

    X = df[features].to_numpy(dtype=float)
    y = df["species_binary"].to_numpy(dtype=int)

    # Load model and predict
    clf = LogisticRegression.load(model_path)
    probs = clf.predict_proba(X)
    ypred = (probs >= threshold).astype(int)

    # Metrics
    acc = accuracy_score(y, ypred)
    pr = precision_recall_f1(y, ypred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pr['precision']:.4f}  Recall: {pr['recall']:.4f}  F1: {pr['f1']:.4f}")

if __name__ == "__main__":
    main()
