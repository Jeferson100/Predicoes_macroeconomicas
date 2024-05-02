from mapie.metrics import (regression_coverage_score,
                           regression_mean_width_score)
from mapie.regression import MapieRegressor

class RegressaoConformal:(model,
    X_train, X_test, y_train, y_test, n_splits,
)
    
    # Estimate prediction intervals on test set with best estimator
    # Here, a non-nested CV approach is used for the sake of computational
    # time, but a nested CV approach is preferred.
    # See the dedicated example in the gallery for more information.
    def regressao_conformal(self, n_splits, alpha=0.05):
        mapie = MapieRegressor(
            model, method="plus", cv=n_splits, agg_function="median", n_jobs=-1
        )
        mapie.fit(self.X_train, self.y_train)
        self.y_pred, self.y_pis = mapie.predict(self.X_test, alpha=alpha)
        coverage = regression_coverage_score(self.y_test, self.y_pis[:, 0, 0], y_pis[:, 1, 0])
        width = regression_mean_width_score(self.y_pis[:, 0, 0], self.y_pis[:, 1, 0])

        # Print results
        print(
            "Coverage and prediction interval width mean for CV+: "
            f"{coverage:.3f}, {width:.3f}"
        )
        return y_pred, y_pis, coverage, width
    def plot_prediction_intervals(self, teste, y_pred, y_pis,legend:str="Colocar a legenda",y_label:str="Colocar o label"):
        # Plot estimated prediction intervals on test set
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_ylabel(y_label)
        ax.plot(teste, lw=2, label="Test data", c="C1")
        ax.plot(teste, y_pred, lw=2, c="C2", label="Predictions")
        ax.fill_between(
            teste.index,
            y_pis[:, 0, 0],
            y_pis[:, 1, 0],
            color="C2",
            alpha=0.2,
            label=legend,
        )
        ax.legend()
        plt.show()