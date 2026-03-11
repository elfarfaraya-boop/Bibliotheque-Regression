
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List

#Dataset — validation et préparation des données
class Dataset:
    def __init__(self, X: np.ndarray, y: np.ndarray, features_names: List[str]):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X et y doivent être des np.ndarray.")

        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X a {X.shape[0]} lignes mais y en a {y.shape[0]}.")

        if np.any(np.isnan(X)) or np.any(np.isnan(y)):
            raise ValueError("Les données contiennent des valeurs manquantes.")

        if len(features_names) != X.shape[1]:
            raise ValueError(
                f"features_names a {len(features_names)} éléments "
                f"mais X a {X.shape[1]} colonnes.")

        self.X = X.copy()           # copie pour ne pas modifier l'array original
        self.y = y.reshape(-1, 1)   # force la forme (n, 1)
        self.features_names = list(features_names)

    def add_intercept(self) -> None:
        if self.features_names and self.features_names[0] == "intercept":
            return

        ones = np.ones((self.X.shape[0], 1))
        self.X = np.hstack((ones, self.X))
        self.features_names.insert(0, "intercept")

    def __repr__(self) -> str:
        n, p = self.X.shape
        return f"Dataset(n={n}, p={p}, features={self.features_names})"

#Régression linéaire — estimation OLS

class LinearRegression:
    def __init__(self):
        self.coefficients: Optional[np.ndarray] = None
        self.features_names: Optional[List[str]] = None
    def fit( self, X: np.ndarray, y: np.ndarray, features_names: Optional[List[str]] = None) -> "LinearRegression":
        y = y.reshape(-1, 1)
        self.coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        self.features_names = features_names
        return self

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        if self.coefficients is None:
            raise ValueError("Modèle non entraîné .")
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        return X_new @ self.coefficients  # forme (n, 1)

    def to_dict(self) -> dict:
        if self.coefficients is None or self.features_names is None:
            raise ValueError("Modèle non entraîné ou features_names manquant.")
        return { name: float(self.coefficients[i, 0])
            for i, name in enumerate(self.features_names)}

    def __repr__(self) -> str:
        status = "entraîné" if self.coefficients is not None else "non entraîné"
        return f"LinearRegression({status})"

#Résultats de la regression— métriques

class RegressionResult:
    def __init__(self, model: LinearRegression, y_true: np.ndarray, y_pred: np.ndarray):
        self.model  = model
        self.y_true = y_true.reshape(-1, 1)
        self.y_pred = y_pred.reshape(-1, 1)

        self.r2   = self._r2()
        self.mse  = self._mse()
        self.rmse = float(np.sqrt(self.mse))

    def _r2(self) -> float:
        ssr = float(np.sum((self.y_true - self.y_pred) ** 2))
        sst = float(np.sum((self.y_true - np.mean(self.y_true)) ** 2))
        if sst == 0:
            return 1.0 if ssr == 0 else 0.0
        return 1.0 - ssr / sst

    def _mse(self) -> float:
        return float(np.mean((self.y_true - self.y_pred) ** 2))

    def summary(self) -> None:
        print("   Régression linéaire OLS — Résultats")

        print("\nCoefficients :")
        df_coef = pd.Series(self.model.to_dict(), name="Valeur").to_frame()
        print(df_coef.to_markdown(numalign="left", stralign="left"))

        print(f"\nR²   : {self.r2:.4f}")
        print(f"MSE  : {self.mse:.4f}")
        print(f"RMSE : {self.rmse:.4f}")

        print("\nAperçu des résidus (5 premières observations) :")
        n = min(5, len(self.y_true))
        rows = [
            {
                "y_true": float(self.y_true[i, 0]),
                "y_pred": round(float(self.y_pred[i, 0]), 4),
                "résidu": round(float(self.y_true[i, 0] - self.y_pred[i, 0]), 4),
            }
            for i in range(n)
        ]
        print(pd.DataFrame(rows).to_markdown(index=False))
        print("=" * 50)

    def __repr__(self) -> str:
        return f"RegressionResult(r2={self.r2:.4f}, mse={self.mse:.4f}, rmse={self.rmse:.4f})"

#Pipeline — fonction principale
def train_linear_regression(dataset: Dataset) -> RegressionResult:
    dataset.add_intercept()

    model = LinearRegression()
    model.fit(dataset.X, dataset.y, features_names=dataset.features_names)

    y_pred = model.predict(dataset.X)
    return RegressionResult(model, dataset.y, y_pred)

if __name__ == "__main__":
    from sklearn.datasets import make_regression

    # --- Exemple 1 : régression simple (1 variable) ---
    print("\nExemple 1 : régression simple")
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=0)
    dataset = Dataset(X, y, features_names=["X1"])
    result = train_linear_regression(dataset)
    result.summary()

    # Visualisation
    plt.figure(figsize=(7, 4))
    plt.scatter(dataset.X[:, 1], dataset.y, alpha=0.6, label="Données réelles")
    plt.plot(dataset.X[:, 1], result.y_pred, color="red", linewidth=2, label="Droite OLS")
    plt.xlabel("X1")
    plt.ylabel("y")
    plt.title(f"Régression simple  |  R² = {result.r2:.4f}")
    plt.legend()
    plt.tight_layout()
    plt.show()