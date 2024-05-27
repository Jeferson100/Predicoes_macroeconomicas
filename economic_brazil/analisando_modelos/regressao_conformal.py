import plotly.graph_objects as go
import plotly.io as pio
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score
from typing import Union, Optional, List, Dict
from typing_extensions import TypedDict
from mapie.quantile_regression import MapieQuantileRegressor
from mapie.subsample import Subsample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, QuantileRegressor


class ConformalRegressionPlotter:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.y_pis = None

    def regressao_conformal(
        self, method="plus", n_splits=5, agg_function="median", n_jobs=-1, alpha=0.05
    ):
        mapie = MapieRegressor(
            self.model,
            method=method,
            cv=n_splits,
            agg_function=agg_function,
            n_jobs=n_jobs,
        )
        mapie.fit(self.X_train, self.y_train)
        self.y_pred, self.y_pis = mapie.predict(self.X_test, alpha=alpha)
        coverage = regression_coverage_score(
            self.y_test, self.y_pis[:, 0, 0], self.y_pis[:, 1, 0]
        )
        width = regression_mean_width_score(self.y_pis[:, 0, 0], self.y_pis[:, 1, 0])

        print(
            f"Coverage and prediction interval width mean for CV+: {coverage:.3f}, {width:.3f}"
        )
        return self.y_pred, self.y_pis, coverage, width

    def plot_prediction_intervals(
        self,
        index_test,
        index_train,
        legend="Prediction Intervals",
        y_label="Response Variable",
        title="Prediction Intervals",
        save=None,
        diretorio=None,
    ):
        if self.y_pred is None or self.y_pis is None:
            raise ValueError(
                "Predictions and intervals not calculated. Please run regressao_conformal first."
            )

        fig = go.Figure()
        # Train data

        fig.add_trace(
            go.Scatter(
                x=index_train,
                y=self.y_train,
                mode="lines",
                name="Train Data",
                line=dict(color="blue"),
            )
        )

        # Test data
        fig.add_trace(
            go.Scatter(
                x=index_test,
                y=self.y_test,
                mode="lines",
                name="Test Data",
                line=dict(color="red"),
            )
        )
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=index_test,
                y=self.y_pred,
                mode="lines",
                name="Predictions",
                line=dict(color="green"),
            )
        )
        # Prediction intervals
        fig.add_trace(
            go.Scatter(
                x=index_test,
                y=self.y_pis[:, 0, 0],
                fill=None,
                mode="lines",
                line=dict(color="lightgrey"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index_test,
                y=self.y_pis[:, 1, 0],
                fill="tonexty",
                mode="lines",
                line=dict(color="lightgrey"),
                name=legend,
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Data",
            yaxis_title=y_label,
            width=2000,  # Width of the figure in pixels
            height=1000,  # Height of the figure in pixels
        )
        if save:
            pio.write_image(fig, diretorio, format="png")
        else:
            fig.show()


class ConformalAvaliandoMetodo:
    def __init__(
        self, X_train, X_test, y_train, y_test, strategies: Optional[Dict] = None
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = {}
        self.y_pis = {}
        if strategies is None:
            self.STRATEGIES = self.default_strategies()
        else:
            self.STRATEGIES = strategies

    def default_strategies(self):
        Params = TypedDict(
            "Params",
            {"method": str, "cv": Union[int, str, Subsample], "alpha": Optional[float]},
        )
        return {
            "naive": Params(method="naive"),
            "jackknife": Params(method="base", cv=-1),
            "jackknife_plus": Params(method="plus", cv=-1),
            "jackknife_minmax": Params(method="minmax", cv=-1),
            "cv": Params(method="base", cv=10),
            "cv_plus": Params(method="plus", cv=10),
            "cv_minmax": Params(method="minmax", cv=10),
            "jackknife_plus_ab": Params(method="plus", cv=Subsample(n_resamplings=50)),
            "jackknife_minmax_ab": Params(
                method="minmax", cv=Subsample(n_resamplings=50)
            ),
            "conformalized_quantile_regression": Params(
                method="quantile", cv="split", alpha=0.05
            ),
        }

    def regressao_conformal_estrategias(self):
        for strategy, params in self.STRATEGIES.items():
            if strategy == "conformalized_quantile_regression":
                quantile_regression = QuantileRegressor(solver="highs", alpha=0)
                mapie = MapieQuantileRegressor(quantile_regression, **params)
                mapie.fit(self.X_train, self.y_train, random_state=1)
                self.y_pred[strategy], self.y_pis[strategy] = mapie.predict(self.X_test)
            else:
                line_model = LinearRegression()
                mapie = MapieRegressor(line_model, **params)
                mapie.fit(self.X_train, self.y_train)
                self.y_pred[strategy], self.y_pis[strategy] = mapie.predict(
                    self.X_test, alpha=0.05
                )
        return self.y_pred, self.y_pis

    def _plot_1d_data(
        self,
        X_train_index,
        y_train,
        X_test_index,
        y_test,
        y_sigma,
        y_pred,
        y_pred_low,
        y_pred_up,
        ax,
        title,
    ):
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.fill_between(X_test_index, y_pred_low, y_pred_up, alpha=0.3)
        ax.scatter(
            X_train_index, y_train, color="red", alpha=0.3, label="Training data"
        )
        ax.plot(X_test_index, y_test, color="gray", label="True confidence intervals")
        ax.plot(X_test_index, y_test - y_sigma, color="gray", ls="--")
        ax.plot(X_test_index, y_test + y_sigma, color="gray", ls="--")
        ax.plot(
            X_test_index, y_pred, color="blue", alpha=0.5, label="Prediction intervals"
        )
        if title is not None:
            ax.set_title(title)
        ax.legend()

    def plotar_metodos_conformal(
        self, index_train, index_teste, strategies: List = None, noise=None
    ):
        if strategies is None:
            strategies = [
                "jackknife_plus",
                "jackknife_minmax",
                "cv_plus",
                "cv_minmax",
                "jackknife_plus_ab",
                "conformalized_quantile_regression",
            ]
        if noise is None:
            noise = np.std(self.y_test)
        _ = len(strategies)
        _, axs = plt.subplots(3, 2, figsize=(15, 20))
        coords = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1], axs[2, 0], axs[2, 1]]
        for strategy, coord in zip(strategies, coords):
            self._plot_1d_data(
                index_train,
                self.y_train,
                index_teste,
                self.y_test,
                np.full((self.y_test.shape[0]), 1.96 * noise),
                self.y_pred[strategy],
                self.y_pis[strategy][:, 0, 0].ravel(),
                self.y_pis[strategy][:, 1, 0].ravel(),
                ax=coord,
                title=strategy,
            )

    def metricas_comparacoes(self):
        return pd.DataFrame(
            [
                [
                    regression_coverage_score(
                        self.y_test,
                        self.y_pis[strategy][:, 0, 0],
                        self.y_pis[strategy][:, 1, 0],
                    ),
                    (
                        self.y_pis[strategy][:, 1, 0] - self.y_pis[strategy][:, 0, 0]
                    ).mean(),
                ]
                for strategy in self.STRATEGIES
            ],
            index=self.STRATEGIES,
            columns=["Coverage", "Width average"],
        ).round(2)
