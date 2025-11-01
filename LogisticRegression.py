"""
LogisticRegression.py
Binary logistic regression using NumPy with batch gradient descent.
Includes standardization parameters for reproducible inference.
"""
from __future__ import annotations
import numpy as np
from typing import Optional
from utils import add_bias, standardize, save_npz, load_npz, poly_features

class LogisticRegression:
    def __init__(self, fit_intercept: bool = True, lr: float = 0.1, epochs: int = 2000, degree: int = 1):
        self.fit_intercept = fit_intercept
        self.lr = lr
        self.epochs = epochs
        self.degree = degree
        self.coef_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -50, 50)
        return 1.0 / (1.0 + np.exp(-z))

    def _prepare(self, X: np.ndarray) -> np.ndarray:
        Xp = poly_features(X, degree=self.degree)
        Xs, mean, std = standardize(Xp)
        self.mean_, self.std_ = mean, std
        if self.fit_intercept:
            Xs = add_bias(Xs)
        return Xs

    def _prepare_infer(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Cannot infer before fitting: feature statistics are missing.")
        Xp = poly_features(X, degree=self.degree)
        Xs, _, _ = standardize(Xp, mean=self.mean_, std=self.std_)
        if self.fit_intercept:
            Xs = add_bias(Xs)
        return Xs

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self._prepare(X)
        n, d = Xs.shape
        w = np.zeros(d)
        for _ in range(self.epochs):
            z = Xs @ w
            p = self._sigmoid(z)
            grad = (1/n) * (Xs.T @ (p - y))
            w -= self.lr * grad
        self.coef_ = w
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("Cannot predict before fitting: coefficients are missing.")
        Xs = self._prepare_infer(X)
        return self._sigmoid(Xs @ self.coef_)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # -------- I/O --------
    def save(self, path: str):
        save_npz(path, coef=self.coef_, mean=self.mean_, std=self.std_, 
                 fit_intercept=np.array(self.fit_intercept), degree=np.array(self.degree))

    @classmethod
    def load(cls, path: str) -> "LogisticRegression":
        data = load_npz(path)
        model = cls(fit_intercept=bool(data["fit_intercept"]), degree=int(data["degree"]))
        model.coef_ = data["coef"]
        model.mean_ = data["mean"]
        model.std_ = data["std"]
        return model
