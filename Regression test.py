
import numpy as np
import pytest
from Regression_pipeline import Dataset, LinearRegression, RegressionResult, train_linear_regression

#Données de test
X_parfait = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_parfait  = np.array([2.0, 4.0, 6.0, 8.0, 10.0])

rng     = np.random.default_rng(42)
X_bruit = rng.standard_normal((100, 2))
y_bruit = 3 * X_bruit[:, 0] - 2 * X_bruit[:, 1] + rng.standard_normal(100) * 0.5

#Tests — Dataset

def test_y_reshape():
    ds = Dataset(X_parfait, y_parfait, features_names=["X1"])
    assert ds.y.shape == (5, 1)

def test_add_intercept_ajoute_colonne():
    ds = Dataset(X_parfait, y_parfait, features_names=["X1"])
    ds.add_intercept()
    assert ds.X.shape[1] == 2
    assert ds.features_names[0] == "intercept"
    assert np.all(ds.X[:, 0] == 1.0)

def test_add_intercept_idempotent():
    ds = Dataset(X_parfait, y_parfait, features_names=["X1"])
    ds.add_intercept()
    ds.add_intercept()
    assert ds.X.shape[1] == 2

def test_erreur_si_dimensions_incompatibles():
    with pytest.raises(ValueError):
        Dataset(np.ones((5, 1)), np.ones(3), features_names=["X1"])


#Tests — LinearRegression

def test_fit_retourne_self():
    ds = Dataset(X_parfait, y_parfait, features_names=["X1"])
    ds.add_intercept()
    model = LinearRegression()
    assert model.fit(ds.X, ds.y) is model

def test_predict_fit_parfait():
    ds = Dataset(X_parfait, y_parfait, features_names=["X1"])
    ds.add_intercept()
    model = LinearRegression()
    model.fit(ds.X, ds.y, features_names=ds.features_names)
    y_pred = model.predict(ds.X)
    np.testing.assert_allclose(y_pred.reshape(-1), y_parfait, atol=1e-6)

def test_predict_vecteur_1d():
    ds = Dataset(X_parfait, y_parfait, features_names=["X1"])
    ds.add_intercept()
    model = LinearRegression()
    model.fit(ds.X, ds.y)
    pred = model.predict(ds.X[0])
    assert pred.shape == (1, 1)

#Tests — RegressionResult
def test_mse_vaut_0_sur_fit_parfait():
    result = train_linear_regression(Dataset(X_parfait, y_parfait, ["X1"]))
    assert result.mse == pytest.approx(0.0, abs=1e-6)

def test_rmse_egal_racine_mse():
    result = train_linear_regression(Dataset(X_bruit, y_bruit, ["X1", "X2"]))
    assert result.rmse == pytest.approx(np.sqrt(result.mse))

def test_r2_positif_sur_donnees_bruitees():
    result = train_linear_regression(Dataset(X_bruit, y_bruit, ["X1", "X2"]))
    assert 0 < result.r2 <= 1.0