import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer

class TimeSeriesModelTuner:
    def __init__(self, model, dados,coluna, numero_divisoes=5,gap_series=5,max_train_size=100,test_size=10):
        self.model = model
        self.X_train = dados.drop(columns=coluna, axis=1)
        self.y_train = dados[coluna]
        self.tscv = TimeSeriesSplit(n_splits=numero_divisoes, gap=gap_series, max_train_size=max_train_size, test_size=test_size)

    def grid_search(self, param_grid, scoring='squared_mean_squared_error', random_state=0):
        grid_search = GridSearchCV(self.model, param_grid, cv=self.tscv, scoring=scoring, random_state=random_state)
        grid_search.fit(self.X_train, self.y_train)
        return grid_search.best_params_, grid_search.best_score_

    def random_search(self, param_distributions, n_iter=10, scoring='squared_mean_squared_error', random_state=0):
        random_search = RandomizedSearchCV(self.model, param_distributions, n_iter=n_iter, cv=self.tscv, scoring=scoring, random_state=random_state)
        random_search.fit(self.X_train, self.y_train)
        return random_search.best_params_, random_search.best_score_

    def bayesian_optimization(self, search_spaces, n_iter=32, scoring='neg_mean_absolute_error', random_state=0):
        bayes_search = BayesSearchCV(self.model, search_spaces, n_iter=n_iter, cv=self.tscv, scoring=scoring, random_state=random_state)
        bayes_search.fit(self.X_train, self.y_train)
        return bayes_search.best_params_, bayes_search.best_score_