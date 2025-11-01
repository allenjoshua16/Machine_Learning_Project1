"""
LinearRegression.py
Implementation aligned with DASC5304 Assignment 1 specification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from utils import add_bias, poly_features, standardize, save_npz, load_npz


@dataclass
class TrainingHistory:
    """Stores per-step losses recorded during training."""
    train_loss: list[float]
    val_loss: list[float]


class LinearRegression:
    def __init__(
        self,
        fit_intercept: bool = True,
        method: str = "gd",
        lr: float = 0.01,
        degree: int = 1,
        validation_split: float = 0.1,
        random_state: Optional[int] = None,
    ) -> None:
        if method != "gd":
            raise ValueError("Only batch gradient descent ('gd') is supported.")
        if not 0 < validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1.")
        self.fit_intercept = bool(fit_intercept)
        self.learning_rate = float(lr)
        self.degree = int(degree)
        self.validation_split = float(validation_split)
        self.random_state = random_state

        self.coef_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.train_loss_history_: list[float] = []
        self.val_loss_history_: list[float] = []
        self.history_: Optional[TrainingHistory] = None

    def _split_train_val(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = X.shape[0]
        val_size = max(1, int(np.ceil(self.validation_split * n_samples)))
        train_size = n_samples - val_size
        if train_size <= 0:
            raise ValueError("Not enough samples for the requested validation split.")

        rng = np.random.default_rng(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)

        val_idx = indices[:val_size]
        train_idx = indices[val_size:]
        return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

    def _prepare_training_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train_raw, X_val_raw, y_train, y_val = self._split_train_val(X, y)

        X_train_poly = poly_features(X_train_raw, degree=self.degree)
        X_val_poly = poly_features(X_val_raw, degree=self.degree)

        X_train_std, mean, std = standardize(X_train_poly)
        X_val_std, _, _ = standardize(X_val_poly, mean=mean, std=std)

        self.mean_ = mean
        self.std_ = std

        if self.fit_intercept:
            X_train_std = add_bias(X_train_std)
            X_val_std = add_bias(X_val_std)

        return X_train_std, X_val_std, y_train, y_val

    @staticmethod
    def _ensure_2d_targets(y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]
        return y_arr

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        regularization: float = 0.0,
        max_epochs: int = 100,
        patience: int = 3,
        shuffle: bool = True,
    ) -> "LinearRegression":
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if patience <= 0:
            raise ValueError("patience must be positive.")
        if regularization < 0:
            raise ValueError("regularization must be non-negative.")

        X_arr = np.asarray(X, dtype=float)
        y_arr = self._ensure_2d_targets(y)

        if X_arr.shape[0] != y_arr.shape[0]:
            raise ValueError("X and y must have the same number of samples.")
        if X_arr.shape[0] < 2:
            raise ValueError("At least two samples are required to fit the model.")

        X_train, X_val, y_train, y_val = self._prepare_training_data(X_arr, y_arr)

        n_train, n_features = X_train.shape
        _, n_outputs = y_train.shape

        coef = np.zeros((n_features, n_outputs), dtype=float)
        reg_mask = np.ones_like(coef)
        if self.fit_intercept:
            reg_mask[0, :] = 0.0

        history_train: list[float] = []
        history_val: list[float] = []

        best_coef = coef.copy()
        best_val_loss = np.inf
        patience_counter = patience

        lr = self.learning_rate
        rng = np.random.default_rng(self.random_state)
        batch_size = min(batch_size, n_train)

        for _ in range(max_epochs):
            if shuffle:
                order = rng.permutation(n_train)
                X_epoch = X_train[order]
                y_epoch = y_train[order]
            else:
                X_epoch = X_train
                y_epoch = y_train

            for start in range(0, n_train, batch_size):
                end = min(start + batch_size, n_train)
                X_batch = X_epoch[start:end]
                y_batch = y_epoch[start:end]
                if X_batch.size == 0:
                    continue

                preds = X_batch @ coef
                errors = preds - y_batch

                grad = (2.0 / X_batch.shape[0]) * (X_batch.T @ errors)
                if regularization:
                    grad += 2.0 * regularization * coef * reg_mask
                coef -= lr * grad

                batch_loss = float(np.mean(errors**2))
                if regularization:
                    batch_loss += float(regularization * np.sum((coef * reg_mask) ** 2))
                history_train.append(batch_loss)

                val_preds = X_val @ coef
                val_errors = val_preds - y_val
                val_loss = float(np.mean(val_errors**2))
                if regularization:
                    val_loss += float(regularization * np.sum((coef * reg_mask) ** 2))
                history_val.append(val_loss)

                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = val_loss
                    best_coef = coef.copy()
                    patience_counter = patience
                else:
                    patience_counter -= 1

                if patience_counter == 0:
                    self.coef_ = best_coef
                    self.train_loss_history_ = history_train
                    self.val_loss_history_ = history_val
                    self.history_ = TrainingHistory(history_train, history_val)
                    return self

        self.coef_ = best_coef
        self.train_loss_history_ = history_train
        self.val_loss_history_ = history_val
        self.history_ = TrainingHistory(history_train, history_val)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.mean_ is None or self.std_ is None:
            raise ValueError("Model must be fitted before calling predict.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        X_poly = poly_features(X_arr, degree=self.degree)
        X_std, _, _ = standardize(X_poly, mean=self.mean_, std=self.std_)
        if self.fit_intercept:
            X_std = add_bias(X_std)

        preds = X_std @ self.coef_
        if preds.shape[1] == 1:
            return preds.ravel()
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_true = self._ensure_2d_targets(y)
        y_pred = self.predict(X)
        if y_pred.ndim == 1:
            y_pred = y_pred[:, None]
        if y_true.shape != y_pred.shape:
            raise ValueError("Shape mismatch between y and predictions.")
        return float(np.mean((y_true - y_pred) ** 2))

    def save(self, path: str) -> None:
        if self.coef_ is None or self.mean_ is None or self.std_ is None:
            raise ValueError("Cannot save an unfitted model.")
        random_state = -1 if self.random_state is None else int(self.random_state)
        save_npz(
            path,
            coef=self.coef_,
            mean=self.mean_,
            std=self.std_,
            fit_intercept=np.array(self.fit_intercept),
            degree=np.array(self.degree),
            learning_rate=np.array(self.learning_rate),
            validation_split=np.array(self.validation_split),
            random_state=np.array(random_state),
        )

    @classmethod
    def load(cls, path: str) -> "LinearRegression":
        data = load_npz(path)

        fit_intercept = bool(np.atleast_1d(data.get("fit_intercept", np.array(True))).item())
        degree = int(np.atleast_1d(data.get("degree", np.array(1))).item())
        lr = float(np.atleast_1d(data.get("learning_rate", np.array(0.01))).item())
        validation_split = float(np.atleast_1d(data.get("validation_split", np.array(0.1))).item())
        random_state_val = int(np.atleast_1d(data.get("random_state", np.array(-1))).item())
        random_state = None if random_state_val < 0 else random_state_val

        model = cls(
            fit_intercept=fit_intercept,
            method="gd",
            lr=lr,
            degree=degree,
            validation_split=validation_split,
            random_state=random_state,
        )

        coef = data.get("coef")
        if coef is None:
            raise ValueError("Model file is missing the 'coef' entry.")
        coef = np.array(coef, dtype=float)
        if coef.ndim == 1:
            coef = coef[:, None]

        mean = data.get("mean")
        std = data.get("std")

        model.coef_ = coef
        model.mean_ = None if mean is None else np.array(mean, dtype=float)
        model.std_ = None if std is None else np.array(std, dtype=float)
        model.train_loss_history_ = []
        model.val_loss_history_ = []
        model.history_ = None
        return model
