# Bibliotheque-Regression
Régression linéaire OLS implémentée from scratch en Python — validation des données, estimation matricielle , métriques R²/MSE/RMSE et tests unitaires.

## Contenu

| Fichier | Description |
|---|---|
| `Regression_pipeline.py` | Bibliothèque principale (Dataset, LinearRegression, RegressionResult, pipeline) |
| `test_regression.py` | Tests unitaires (pytest) |

## Installation
```bash
pip install numpy pandas matplotlib pytest
```

## Utilisation rapide
```python
import numpy as np
from Regression_pipeline import Dataset, train_linear_regression

X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

dataset = Dataset(X, y, features_names=["X1"])
result  = train_linear_regression(dataset)
result.summary()
```

## Structure du code

- **`Dataset`** — encapsule et valide les données (détection NaN, dimensions, ajout intercept)
- **`LinearRegression`** — estime les coefficients via `np.linalg.lstsq` (stable en cas de multicolinéarité)
- **`RegressionResult`** — calcule et affiche R², MSE, RMSE et les résidus
- **`train_linear_regression()`** — pipeline complet en une ligne

## Lancer les tests
```bash
pytest test_regression.py -v
```

## Choix techniques

- `np.linalg.lstsq` plutôt que `np.linalg.inv` pour la stabilité numérique sous multicolinéarité
- Métriques calculées une seule fois à l'instanciation (attributs, pas méthodes)
- `add_intercept()` idempotente : sans risque d'appel multiple
