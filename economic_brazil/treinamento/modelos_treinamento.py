import pandas as pd
#import os modelos de regressao
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor


class TreinamentoModelos:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def treinar_gradient_boosting(self, max_depth=3, n_estimators=100, learning_rate=0.1, random_state=0):
        regre_gb = GradientBoostingRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        regre_gb.fit(self.X, self.y)
        return regre_gb
    
    def treinar_xgboost(self, max_depth=3, n_estimators=100, learning_rate=0.1, random_state=0):
        regre_xgb = XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        regre_xgb.fit(self.X, self.y)
        return regre_xgb