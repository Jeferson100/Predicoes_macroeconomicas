from typing import Any, Dict
from economic_brazil.treinamento.modelos_treinamento import TreinamentoModelos
from economic_brazil.treinamento.redes_neurais_recorrentes import (
    RnnModel,
    HyperTurnerModel,
)
from economic_brazil.treinamento.arima_treinamento import Sarimax
from economic_brazil.treinamento.treinamento_modelos_tuning import TimeSeriesModelTuner
import joblib
import pickle
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
# pylint: disable=E0401
from keras.models import load_model
from keras.losses import MeanSquaredError
# pylint: disable=E0401

class TreinandoModelos:
    def __init__(
        self,
        x_treino,
        y_treino,
        x_teste,
        y_teste,
        tuning_grid_search=False,
        tuning_random_search=False,
        tuning_bayes_search=True,
        salvar_modelo=False,
        diretorio=None,
    ):
        self.x_treino = x_treino
        self.y_treino = y_treino
        self.x_teste = x_teste
        self.y_teste = y_teste
        self.modelos = TreinamentoModelos(self.x_treino, self.y_treino)
        self.neural_network = RnnModel()
        self.sarimax = Sarimax()
        self.tuning_grid_search = tuning_grid_search
        self.tuning_random_search = tuning_random_search
        self.tuning_bayes_search = tuning_bayes_search
        self.salvar_modelo = salvar_modelo
        self.diretorio = diretorio

    def treinar_modelos(
        self,
        gradiente_boosting=True,
        xg_boost=True,
        cat_boost=True,
        regressao_linear=True,
        redes_neurais=True,
        redes_neurais_tuning=None,
        sarimax=True,
        tuning_sarimax=None,
        param_grid_gradiente=None,
        param_grid_xgboost=None,
        param_grid_catboost=None,
    ) -> Dict[str, Any]:
        """
        Treina vários modelos de machine learning com a possibilidade de tunagem de hiperparâmetros.

        :return: Dicionário com os modelos treinados.
        """
        resultados = {}

        if gradiente_boosting:
            if param_grid_gradiente is None:
                param_grid_gradiente = {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                    "max_depth": [1, 3, 5, 7, 9],
                }
            modelo_gradiente = self._treinar_com_tunagem(
                self.modelos.modelo_gradient_boosting(),
                self.modelos.treinar_gradient_boosting,
                param_grid_gradiente,
            )
            resultados["gradiente_boosting"] = modelo_gradiente
            print("Modelo Gradiente Boosting Treinado")

        if xg_boost:
            if param_grid_xgboost is None:
                param_grid_xgboost = {
                    "n_estimators": [100, 200, 300, 400, 500],
                    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                    "max_depth": [1, 3, 5, 7, 9],
                }
            modelo_xgboost = self._treinar_com_tunagem(
                self.modelos.modelo_xgboost(),
                self.modelos.treinar_xgboost,
                param_grid_xgboost,
            )
            resultados["xg_boost"] = modelo_xgboost
            print("Modelo XG Boost Treinado")

        if cat_boost:
            if param_grid_catboost is None:
                param_grid_catboost = {
                    "iterations": [1, 2, 10, 50, 100],
                    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                    "depth": [1, 3, 5, 7, 9],
                }
            modelo_catboost = self._treinar_com_tunagem(
                self.modelos.modelo_catboost(),
                self.modelos.treinar_catboost,
                param_grid_catboost,
            )
            resultados["cat_boost"] = modelo_catboost
            print("Modelo Cat Boost Treinado")

        if regressao_linear:
            modelo_regressao_linear = self.modelos.treinar_regressao_linear()
            resultados["regressao_linear"] = modelo_regressao_linear
            print("Modelo Regressão Linear Treinado")

        if redes_neurais:
            modelo_redes_neurais = self.redes_neurais(
                redes_neurais_tuning=redes_neurais_tuning
            )
            resultados["redes_neurais"] = modelo_redes_neurais
            print("Modelo Redes Neurais Treinado")

        if sarimax:
            modelo_sarimax = self.treinar_sarimax(tuning_sarimax=tuning_sarimax)
            resultados["sarimax"] = modelo_sarimax

        if self.salvar_modelo:
            self.salvar(
                diretorio=self.diretorio,
                resultados=resultados,
                gradiente_boosting=gradiente_boosting,
                xg_boost=xg_boost,
                cat_boost=cat_boost,
                regressao_linear=regressao_linear,
                redes_neurais=redes_neurais,
                sarimax=sarimax,
            )

        return resultados

    def _treinar_com_tunagem(
        self,
        modelo,
        treinar_func,
        param_grid,
    ) -> Any:
        """
        Função auxiliar para treinar o modelo com tunagem de hiperparâmetros.

        :return: Modelo treinado.
        """
        tuning = TimeSeriesModelTuner(modelo, self.x_treino, self.y_treino)

        if self.tuning_grid_search:
            best_params, _ = tuning.grid_search(param_grid)
        elif self.tuning_random_search:
            best_params, _ = tuning.random_search(param_grid)
        elif self.tuning_bayes_search:
            best_params, _ = tuning.bayesian_optimization(param_grid)
        else:
            best_params = {}

        return treinar_func(**best_params)

    def redes_neurais(self, redes_neurais_tuning=None):
        """
        Treina uma rede neural recorrente (RNN) com os dados fornecidos.

        :return: Modelo RNN treinado.
        """
        x_train_neural, y_train_neural = self.neural_network.create_dataset(
            self.x_treino, self.y_treino
        )
        x_teste_neural, y_teste_neural = self.neural_network.create_dataset(
            self.x_teste, self.y_teste
        )
        if redes_neurais_tuning:
            tunning_neural = HyperTurnerModel(
                x_train_neural, y_train_neural, x_teste_neural, y_teste_neural
            )
            model_neural = tunning_neural.tuner(epochs=10)
        else:
            model_neural = self.neural_network.create_rnn_model(
                input_shape=(x_train_neural.shape[1], x_train_neural.shape[2])
            )
            model_neural = self.neural_network.treinar_modelo(
                model_neural,
                x_train_neural,
                y_train_neural,
                test_data=True,
                X_test=x_teste_neural,
                y_test=y_teste_neural,
            )

        return model_neural

    def treinar_sarimax(self, tuning_sarimax=None):
        """
        Treina o modelo SARIMAX com os dados fornecidos.

        :return: modelos SARIMAX treinado.
        """
        if tuning_sarimax:
            _, valor_p, valor_q = self.sarimax.encontrar_parametros_sarimax(
                self.x_treino, self.y_treino
            )
            modelo_sarimax = self.sarimax.treinar_sarimax(
                self.y_treino, self.x_treino, p=valor_p, d=0, q=valor_q
            )

        else:
            modelo_sarimax = self.sarimax.treinar_sarimax(self.y_treino, self.x_treino)

        return modelo_sarimax

    def salvar(
        self,
        diretorio,
        resultados,
        gradiente_boosting=False,
        xg_boost=False,
        cat_boost=False,
        regressao_linear=False,
        redes_neurais=False,
        sarimax=False,
    ):
        if gradiente_boosting:
            joblib.dump(
                resultados["gradiente_boosting"],
                diretorio + "gradiente_boosting_model.pkl",
            )

        if xg_boost:
            resultados["xg_boost"].save_model(diretorio + "xg_boost_model.json")

        if cat_boost:
            resultados["cat_boost"].save_model(diretorio + "cat_boost_model.cbm")

        if regressao_linear:
            joblib.dump(
                resultados["regressao_linear"], diretorio + "regressao_linear_model.pkl"
            )

        if redes_neurais:
            resultados["redes_neurais"].save(diretorio + "redes_neurais_model.h5")

        if sarimax:
            with open(diretorio + "sarimax_model.pkl", "wb") as f:
                pickle.dump(resultados["sarimax"], f)


def carregar(
    diretorio,
    gradiente_boosting=True,
    xg_boost=True,
    cat_boost=True,
    regressao_linear=True,
    redes_neurais=True,
    sarimax=True,
):
    resultados = {}

    if gradiente_boosting:
        resultados["gradiente_boosting"] = joblib.load(
            diretorio + "gradiente_boosting_model.pkl"
        )

    if xg_boost:
        resultados["xg_boost"] = XGBRegressor()
        resultados["xg_boost"].load_model(diretorio + "xg_boost_model.json")

    if cat_boost:
        resultados["cat_boost"] = CatBoostRegressor()
        resultados["cat_boost"].load_model(diretorio + "cat_boost_model.cbm")

    if regressao_linear:
        resultados["regressao_linear"] = joblib.load(
            diretorio + "regressao_linear_model.pkl"
        )

    if redes_neurais:
        resultados["redes_neurais"] = load_model(
            diretorio + "redes_neurais_model.h5",
            custom_objects={"mse": MeanSquaredError()},
        )

    if sarimax:
        with open(diretorio + "sarimax_model.pkl", "rb") as f:
            resultados["sarimax"] = pickle.load(f)

    return resultados
