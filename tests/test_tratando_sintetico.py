"""Tratamento de dados sem rede.

O objetivo destes testes e travar os bugs que geraram as previsoes de out/26:
features de teste em escala diferente do treino, alvo preenchido por ffill e
selecao ajustada no conjunto errado.
"""

import numpy as np
import pandas as pd
import pytest

from economic_brazil.processando_dados.tratando_dados import TratandoDados


def _tratando(dados: pd.DataFrame, **kwargs) -> TratandoDados:
    parametros = {
        "coluna_label": "selic",
        "data_divisao": "2018-01-01",
        "numero_defasagens": 2,
        "covid": False,
        "datas": False,
        "pca": False,
        "variancia": False,
        "smart_correlation": False,
        "rfe": False,
    }
    parametros.update(kwargs)
    return TratandoDados(dados, **parametros)


def test_treino_e_teste_tem_as_mesmas_colunas(dados_sinteticos):
    defeito = _tratando(dados_sinteticos, estacionaridade=False)
    x_treino, x_teste, y_treino, y_teste, *_ = defeito.tratando_dados()

    assert x_treino.shape[1] == x_teste.shape[1]
    assert len(y_treino) == len(x_treino)
    assert len(y_teste) == len(x_teste)


def test_estacionaridade_ajustada_no_treino_nao_reajusta_no_teste(dados_sinteticos):
    tratando = _tratando(dados_sinteticos, estacionaridade=True)
    tratando.tratando_dados()

    assert tratando.estacionaridade_modelo is not None

    # o transform usa o modelo ajustado: nao aceita ser chamado sem fit
    novo = _tratando(dados_sinteticos, estacionaridade=True)
    with pytest.raises(RuntimeError):
        novo.tratando_estacionaridade(dados_sinteticos, coluna_label="selic", fit=False)


def test_sem_nan_no_conjunto_de_treino(dados_sinteticos):
    tratando = _tratando(dados_sinteticos, estacionaridade=False)
    x_treino, x_teste, y_treino, y_teste, *_ = tratando.tratando_dados()

    assert not np.isnan(np.asarray(x_treino, dtype=float)).any()
    assert not np.isnan(np.asarray(x_teste, dtype=float)).any()
    assert not np.isnan(np.asarray(y_treino, dtype=float)).any()
    assert not np.isnan(np.asarray(y_teste, dtype=float)).any()


def test_alvo_ausente_falha_com_mensagem_clara(dados_sinteticos):
    tratando = _tratando(dados_sinteticos, coluna_label="variavel_inexistente")

    with pytest.raises(KeyError, match="ausente"):
        tratando.tratando_dados()
