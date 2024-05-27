import sys

sys.path.append("..")
from economic_brazil.processando_dados.divisao_treino_teste import (
    treino_test_dados,
    treino_teste_seies_temporal,
)
import pandas as pd
import numpy as np
import pytest


@pytest.fixture
# pylint: disable=W0621
def sample_data():
    # Dados de exemplo para os testes
    dates = pd.date_range(start="2020-01-01", periods=20, freq="D")
    data = {"A": np.arange(20), "B": np.random.rand(20)}
    df = pd.DataFrame(data, index=dates)
    return df


# pylint: disable=W0621


def test_treino_test_dados(sample_data):
    # Teste básico para a função treino_test_dados
    treino, teste = treino_test_dados(sample_data, "2020-01-11", treino_teste=True)
    assert treino.shape[0] == 10, "O conjunto de treino deve ter 10 linhas"
    assert teste.shape[0] == 10, "O conjunto de teste deve ter 10 linhas"

    # pylint: disable=W0632
    x_treino, y_treino, x_teste, y_teste = treino_test_dados(
        sample_data, "2020-01-11", coluna="A", treino_teste=False
    )
    # pylint: disable=W0632
    assert y_treino.shape[0] == 10, "O conjunto y_treino deve ter 10 linhas"
    assert x_treino.shape[0] == 10, "O conjunto x_treino deve ter 10 linhas"
    assert y_teste.shape[0] == 10, "O conjunto y_teste deve ter 10 linhas"
    assert x_teste.shape[0] == 10, "O conjunto x_teste deve ter 10 linhas"


def test_treino_teste_seies_temporal(sample_data):
    # Dados de treino
    x_treino = sample_data.drop(columns=["A"]).values
    y_treino = sample_data["A"].values

    # Teste básico para a função treino_teste_seies_temporal
    splits = treino_teste_seies_temporal(
        x_treino,
        y_treino,
        numero_divisoes=5,
        gap_series=2,
        max_train_size=15,
        test_size=3,
    )
    assert len(splits) == 5, "O número de divisões deve ser 5"

    # Verifique o tamanho dos splits
    for train_index, test_index in splits:
        assert len(test_index) == 3, "O tamanho do conjunto de teste deve ser 3"
        assert (
            len(train_index) <= 15
        ), "O tamanho do conjunto de treino deve ser no máximo 15"

        # Verifique se há uma lacuna entre treino e teste
        assert (
            train_index[-1] < test_index[0] - 2
        ), "Deve haver uma lacuna de 2 dias entre treino e teste"

        # Verifique se os índices estão dentro do tamanho esperado dos dados
        assert max(train_index) < len(
            x_treino
        ), "Índice de treino fora do intervalo dos dados"
        assert max(test_index) < len(
            x_treino
        ), "Índice de teste fora do intervalo dos dados"
