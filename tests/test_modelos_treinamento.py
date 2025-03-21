import sys

sys.path.append("..")
import pytest
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from economic_brazil.treinamento.modelos_treinamento import TreinamentoModelos
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn import linear_model

# pylint: disable=W0621
@pytest.fixture(scope="module")
# pylint: disable=W0632
def data_regressao() -> tuple:
    # Gera um dataset de regressão sintético para testes
    # pylint: disable=W0632
    X, y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=42)  # type: ignore
    # pylint: disable=W0632
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    modelos = TreinamentoModelos(X_train, y_train)
    return X_train, X_test, y_train, y_test, modelos


# pylint: disable=W0621
def test_modelo_gradient_boosting(data_regressao: tuple) -> None:
    _, _, _, _, modelos = data_regressao
    model = modelos.modelo_gradient_boosting()
    assert isinstance(model, GradientBoostingRegressor)


def test_modelo_xgboost(data_regressao: tuple) -> None:
    _, _, _, _, modelos = data_regressao
    model = modelos.modelo_xgboost()
    assert isinstance(model, XGBRegressor)


def test_modelo_catboost(data_regressao: tuple) -> None:
    _, _, _, _, modelos = data_regressao
    model = modelos.modelo_catboost()
    assert isinstance(model, CatBoostRegressor)


def test_modelo_regressao_linear(data_regressao: tuple) -> None:
    _, _, _, _, modelos = data_regressao
    model = modelos.modelo_regressao_linear()
    assert isinstance(model, linear_model.LinearRegression)


def test_treinar_gradient_boosting(data_regressao: tuple) -> None:
    _, X_test, _, y_test, modelos = data_regressao
    model = modelos.treinar_gradient_boosting()
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_treinar_xgboost(data_regressao: tuple) -> None:
    _, X_test, _, y_test, modelos = data_regressao
    model = modelos.treinar_xgboost()
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_treinar_catboost(data_regressao: tuple) -> None:
    _, X_test, _, y_test, modelos = data_regressao
    model = modelos.treinar_catboost()
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_treinar_regressao_linear(data_regressao: tuple) -> None:
    _, X_test, _, y_test, modelos = data_regressao
    model = modelos.treinar_regressao_linear()
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)
