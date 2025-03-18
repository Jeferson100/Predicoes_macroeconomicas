# import os modelos de regressao
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn import linear_model
import pandas as pd
from typing import Optional


class TreinamentoModelos:
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y

    def modelo_gradient_boosting(
        self,
        max_depth: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> GradientBoostingRegressor:
        return GradientBoostingRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def modelo_xgboost(
        self,
        max_depth: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> XGBRegressor:
        return XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def modelo_catboost(
        self, iterations: int = 2, learning_rate: float = 1, depth: int = 2
    ) -> CatBoostRegressor:
        return CatBoostRegressor(
            iterations=iterations, learning_rate=learning_rate, depth=depth
        )

    def modelo_regressao_linear(
        self, copy_X: bool = True, n_jobs: Optional[int] = None
    ):
        return linear_model.LinearRegression(copy_X=copy_X, n_jobs=n_jobs)

    def treinar_gradient_boosting(
        self,
        max_depth: int = 3,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        random_state: int = 0,
    ) -> GradientBoostingRegressor:
        regre_gb = self.modelo_gradient_boosting(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        regre_gb.fit(self.X, self.y)
        return regre_gb

    def treinar_xgboost(
        self, max_depth: int = 3, n_estimators: int = 100, learning_rate: float = 0.1
    ) -> XGBRegressor:
        regre_xgb = self.modelo_xgboost(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        regre_xgb.fit(self.X, self.y)
        return regre_xgb

    def treinar_catboost(
        self, iterations: int = 2, learning_rate: float = 1, depth: int = 2
    ) -> CatBoostRegressor:
        regre_cb = self.modelo_catboost(
            iterations=iterations, learning_rate=learning_rate, depth=depth
        )
        regre_cb.fit(self.X, self.y)
        return regre_cb

    def treinar_regressao_linear(
        self, copy_X: bool = True, n_jobs: Optional[int] = None
    ) -> linear_model.LinearRegression:
        regre_lr = self.modelo_regressao_linear(copy_X=copy_X, n_jobs=n_jobs)
        regre_lr.fit(self.X, self.y)
        return regre_lr
