#!/usr/bin/env python3
"""
Train Classifier (variant 3)
- Binary logistic regression
- Hard-coded CSV path and all four features (sepal_length, sepal_width, petal_length, petal_width)
- Creates a binary target: Iris-versicolor = 1, others = 0
- Trains, prints metrics, and saves model to model_cls3.npz
- Plots decision regions in 2D via PCA projection of the training set
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import train_test_split, accuracy_score, precision_recall_f1
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions

def pca_fit_transform(X, k=2):
    """Return (X_pc, mean, components) where components has shape (d, k)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    # covariance in feature space
    C = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    # eigen-decomposition
    vals, vecs = np.linalg.eigh(C)            # vecs columns = eigenvectors
    idx = np.argsort(vals)[::-1][:k]          # top-k
    components = vecs[:, idx]                 # (d, k)
    X_pc = Xc @ components                    # (n, k)
    return X_pc, mean, components

class PCADecisionWrapper:
    """
    Wraps a trained high-D classifier so plot_decision_regions (2D) can call predict.
    Maps 2D PCA coords -> approx 4D via inverse transform, then uses clf.predict.
    """
    def __init__(self, clf, mean, components):
        self.clf = clf
        self.mean = mean
        self.components = components  # (d, 2)

    def predict(self, X_2d):
        # inverse PCA (rank-2 reconstruction)
        X_recon = X_2d @ self.components.T + self.mean
        return self.clf.predict(X_recon)

def main():
    # === Hard-coded CSV file (edit path if needed) ===
    data_path = r"C:/Users/allen/Downloads/archive (1)/IRIS.csv"

    # === Features for classifier3 (all four features) ===
    features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

    # --- Load data ---
    df = pd.read_csv(data_path)

    # Create binary target: versicolor = 1, others = 0
    species_lower = df["species"].astype(str).str.strip().str.lower()
    df["species_binary"] = (species_lower.str.contains("versicolor")).astype(int)

    # Build X, y as numpy arrays
    X = df[features].to_numpy(dtype=float)
    y = df["species_binary"].to_numpy(dtype=int)

    # Train/validation split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2)

    # Train logistic regression (4D)
    clf = LogisticRegression(lr=0.1, epochs=2000, degree=1)
    clf.fit(Xtr, ytr)

    # Evaluate on held-out test
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    pr = precision_recall_f1(yte, ypred)
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {pr['precision']:.4f}  Recall: {pr['recall']:.4f}  F1: {pr['f1']:.4f}")

    # Save model
    clf.save("model_cls3.npz")
    print("Saved model to model_cls3.npz")

    # === PCA projection (train set) to 2D for visualization ===
    Xtr_pc, mean4d, comps4d_2 = pca_fit_transform(Xtr, k=2)

    # Wrap classifier for 2D plotting (uses inverse PCA to call 4D model)
    clf_wrap = PCADecisionWrapper(clf, mean=mean4d, components=comps4d_2)

    # Plot decision regions in PCA space
    plt.figure(figsize=(8, 6))
    plot_decision_regions(X=Xtr_pc.astype(np.float64), y=ytr.astype(np.int32), clf=clf_wrap, legend=2)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Decision Regions (Logistic Regression - Classifier 3, PCA 2D)")
    plt.tight_layout()
    plt.savefig("classifier3_decision_regions_pca.png", dpi=300)
    plt.show()
    print("Saved plot to classifier3_decision_regions_pca.png")

if __name__ == "__main__":
    main()
