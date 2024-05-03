import plotly.graph_objects as go
import plotly.io as pio
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score, regression_mean_width_score

class ConformalRegressionPlotter:
    def __init__(self, model, X_train, X_test, y_train, y_test):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = None
        self.y_pis = None

    def regressao_conformal(self, method="plus", n_splits=5, agg_function="median", n_jobs=-1, alpha=0.05):
        mapie = MapieRegressor(self.model, method=method, cv=n_splits, agg_function=agg_function, n_jobs=n_jobs)
        mapie.fit(self.X_train, self.y_train)
        self.y_pred, self.y_pis = mapie.predict(self.X_test, alpha=alpha)
        coverage = regression_coverage_score(self.y_test, self.y_pis[:, 0, 0], self.y_pis[:, 1, 0])
        width = regression_mean_width_score(self.y_pis[:, 0, 0], self.y_pis[:, 1, 0])

        print(f"Coverage and prediction interval width mean for CV+: {coverage:.3f}, {width:.3f}")
        return self.y_pred, self.y_pis, coverage, width

    def plot_prediction_intervals(self, index_train, index_test, legend="Prediction Intervals", y_label="Response Variable",title="Prediction Intervals",save=None,diretorio=None):
        if self.y_pred is None or self.y_pis is None:
            raise ValueError("Predictions and intervals not calculated. Please run regressao_conformal first.")

        fig = go.Figure()
        # Train data
        fig.add_trace(go.Scatter(x=index_train, y=self.y_train, mode='lines', name='Train Data', line=dict(color='blue')))
        # Test data
        fig.add_trace(go.Scatter(x=index_test, y=self.y_test, mode='lines', name='Test Data', line=dict(color='red')))
        # Predictions
        fig.add_trace(go.Scatter(x=index_test, y=self.y_pred, mode='lines', name='Predictions', line=dict(color='green')))
        # Prediction intervals
        fig.add_trace(go.Scatter(x=index_test, y=self.y_pis[:, 0, 0], fill=None, mode='lines', line=dict(color='lightgrey'), showlegend=False))
        fig.add_trace(go.Scatter(x=index_test, y=self.y_pis[:, 1, 0], fill='tonexty', mode='lines', line=dict(color='lightgrey'), name=legend))
        
        fig.update_layout(title=title, xaxis_title='Data', yaxis_title=y_label,width=2000,  # Width of the figure in pixels
            height=1000   # Height of the figure in pixels
        )
        if save:
            pio.write_image(fig, diretorio, format='png')
        else:
            fig.show()