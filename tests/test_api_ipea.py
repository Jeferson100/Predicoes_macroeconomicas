"""Cliente do IPEADATA sem rede."""

import numpy as np
import pandas as pd
import pytest
import requests

from economic_brazil.coleta_dados.configuracao_apis import api_ipea


class _Resposta:
    def __init__(self, valores, status=200):
        self._valores = valores
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return {"value": self._valores}


class _Sessao:
    def __init__(self, *respostas):
        self.respostas = list(respostas)
        self.chamadas = []

    def get(self, url, timeout=None):
        self.chamadas.append((url, timeout))
        resposta = self.respostas.pop(0)
        if isinstance(resposta, Exception):
            raise resposta
        return resposta


def _valores():
    return [
        {"SERCODIGO": "X", "VALDATA": "2020-01-01T00:00:00-02:00", "VALVALOR": 1.0},
        {"SERCODIGO": "X", "VALDATA": "2020-02-01T00:00:00-03:00", "VALVALOR": 2.0},
    ]


def _instalar(monkeypatch, sessao):
    monkeypatch.setattr(api_ipea, "_sessao", lambda: sessao)
    monkeypatch.setattr(api_ipea.time, "sleep", lambda _segundos: None)
    return sessao


def test_parse_converte_data_e_valor(monkeypatch):
    _instalar(monkeypatch, _Sessao(_Resposta(_valores())))

    serie = api_ipea.buscar_serie("X")

    assert isinstance(serie.index, pd.DatetimeIndex)
    assert list(serie.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")]
    assert serie.tolist() == [1.0, 2.0]
    assert serie.name == "X"


def test_data_duplicada_mantem_a_ultima(monkeypatch):
    valores = _valores() + [
        {"SERCODIGO": "X", "VALDATA": "2020-02-01T00:00:00-03:00", "VALVALOR": 9.0}
    ]
    _instalar(monkeypatch, _Sessao(_Resposta(valores)))

    serie = api_ipea.buscar_serie("X")

    assert serie.loc["2020-02-01"] == 9.0
    assert not serie.index.has_duplicates


def test_valor_nulo_vira_nan(monkeypatch):
    valores = [{"SERCODIGO": "X", "VALDATA": "2020-01-01T00:00:00-02:00", "VALVALOR": None}]
    _instalar(monkeypatch, _Sessao(_Resposta(valores)))

    serie = api_ipea.buscar_serie("X")

    assert np.isnan(serie.iloc[0])


def test_serie_vazia_falha(monkeypatch):
    _instalar(monkeypatch, _Sessao(_Resposta([]), _Resposta([]), _Resposta([])))

    with pytest.raises(RuntimeError, match="X"):
        api_ipea.buscar_serie("X")


def test_tenta_de_novo_apos_erro_de_rede(monkeypatch):
    sessao = _instalar(
        monkeypatch,
        _Sessao(requests.ConnectionError("caiu"), _Resposta(_valores())),
    )

    serie = api_ipea.buscar_serie("X")

    assert len(serie) == 2
    assert len(sessao.chamadas) == 2


def test_usa_timeout_explicito(monkeypatch):
    sessao = _instalar(monkeypatch, _Sessao(_Resposta(_valores())))

    api_ipea.buscar_serie("X")

    _, timeout = sessao.chamadas[0]
    assert timeout == api_ipea.TIMEOUT_SEGUNDOS


def test_sessao_unica_e_reaproveitada():
    """O ganho depende de nao abrir conexao nova a cada serie."""
    assert api_ipea._sessao() is api_ipea._sessao()
