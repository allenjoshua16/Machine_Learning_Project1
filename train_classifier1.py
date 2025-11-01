#!/usr/bin/env python3
"""
Train Classifier (variant 1)
- Binary logistic regression
- Hard-coded CSV path and features (petal_length, petal_width)
- Creates a binary target: Iris-versicolor = 1, others = 0
- Trains, prints metrics, saves model to model_cls1.npz
- Plots decision regions (training data) and highlights test points
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import train_test_split, accuracy_score, precision_recall_f1
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions

def main():
    # === Hard-coded CSV file ===
    data_path = r"C:/Users/allen/Downloads/archive (1)/IRIS.csv"

    # === Features for classifier1 (petal length & petal width) ===
    features = ["petal_length", "petal_width"]

    # --- Load data ---
    df = pd.read_csv(data_path)

    # Ensure required columns exist
    missing = [c for c in features + ["species"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Create binary target: versicolor = 1, others = 0 (robust to case/whitespace)
    species_lower = df["species"].astype(str).str.strip().str.lower()
    df["species_binary"] = (species_lower.str.contains("versicolor")).astype(int)

    # Build X, y as numpy arrays
    X = df[features].to_numpy(dtype=float)
    y = df["species_binary"].to_numpy(dtype=int)

    # Train/validation split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

    # Train logistic regression
    clf = LogisticRegression(lr=0.1, epochs=2000, degree=1)
    clf.fit(Xtr, ytr)

    # Evaluate
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    pr = precision_recall_f1(yte, ypred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pr['precision']:.4f}  Recall: {pr['recall']:.4f}  F1: {pr['f1']:.4f}")

    # Save model
    clf.save("model_cls1.npz")
    print("Saved model to model_cls1.npz")

    # === Decision-region plot (train set), with test points highlighted ===
    plt.figure(figsize=(8, 6))
    plot_decision_regions(
        X=Xtr.astype(np.float64),
        y=ytr.astype(np.int32),
        clf=clf,
        X_highlight=Xte.astype(np.float64),  # show test points as hollow markers
        legend=2
    )
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("Decision Regions (Logistic Regression — Classifier 1)")
    plt.tight_layout()
    plt.savefig("classifier1_decision_regions.png", dpi=300)
    plt.show()
    print("Saved plot to classifier1_decision_regions.png")

if __name__ == "__main__":
    main()
