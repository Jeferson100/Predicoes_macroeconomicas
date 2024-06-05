import pytest
import numpy as np
from economic_brazil.treinamento.treinamento_algoritimos import (
    TreinandoModelos,
    carregar,
)


@pytest.fixture
# pylint: disable=W0621
def sample_data():
    # Cria dados de exemplo para teste
    n_samples = 200
    x_treino = np.random.rand(n_samples, 10)
    y_treino = np.random.rand(n_samples)
    x_teste = np.random.rand(n_samples, 10)
    y_teste = np.random.rand(n_samples)
    return x_treino, y_treino, x_teste, y_teste


# pylint: disable=W0621


@pytest.fixture
def data_processor(sample_data):
    x_treino, y_treino, x_teste, y_teste = sample_data
    return TreinandoModelos(
        x_treino=x_treino,
        y_treino=y_treino,
        x_teste=x_teste,
        y_teste=y_teste,
        tuning_grid_search=False,
        tuning_random_search=False,
        tuning_bayes_search=True,
        salvar_modelo=False,
        diretorio=None,
    )


def test_treinar_modelos(data_processor):
    resultados = data_processor.treinar_modelos(
        gradiente_boosting=True,
        xg_boost=True,
        cat_boost=True,
        regressao_linear=True,
        redes_neurais=False,  # Desativado para teste simplificado
        sarimax=False,
    )
    assert "gradiente_boosting" in resultados
    assert "xg_boost" in resultados
    assert "cat_boost" in resultados
    assert "regressao_linear" in resultados


def test_redes_neurais(data_processor):
    model_neural = data_processor.redes_neurais(redes_neurais_tuning=False)
    assert model_neural is not None


def test_treinar_sarimax(data_processor):
    model_sarimax = data_processor.treinar_sarimax(tuning_sarimax=False)
    assert model_sarimax is not None


def test_salvar_e_carregar_modelo(data_processor, tmpdir):
    resultados = data_processor.treinar_modelos(
        gradiente_boosting=True,
        xg_boost=True,
        cat_boost=True,
        regressao_linear=True,
        redes_neurais=False,  # Desativado para teste simplificado
        sarimax=False,
    )
    diretorio = tmpdir.mkdir("modelos")
    data_processor.salvar(
        diretorio=str(diretorio) + "/",
        resultados=resultados,
        gradiente_boosting=True,
        xg_boost=True,
        cat_boost=True,
        regressao_linear=True,
        redes_neurais=False,
        sarimax=False,
    )
    modelos_carregados = carregar(
        diretorio=str(diretorio) + "/",
        gradiente_boosting=True,
        xg_boost=True,
        cat_boost=True,
        regressao_linear=True,
        redes_neurais=False,
        sarimax=False,
    )
    assert "gradiente_boosting" in modelos_carregados
    assert "xg_boost" in modelos_carregados
    assert "cat_boost" in modelos_carregados
    assert "regressao_linear" in modelos_carregados
