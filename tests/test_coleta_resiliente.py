"""Coleta tolerante a falhas: uma fonte quebrada nao derruba o job.

Os testes usam apenas stdlib, pandas e monkeypatch: nenhuma chamada de rede.
"""

import asyncio
import io

import pandas as pd
import pytest

from economic_brazil.coleta_dados.configuracao_apis.api_fred import set_fred_api_key
from economic_brazil.coleta_dados.configuracao_apis.api_quandl import set_quandl_api_key
from economic_brazil.coleta_dados.economic_data_brazil_async import EconomicBrazilAsync


class _SemTerminal(io.StringIO):
    def isatty(self) -> bool:
        return False


@pytest.fixture
def sem_terminal(monkeypatch):
    monkeypatch.setattr("sys.stdin", _SemTerminal())


@pytest.fixture
def coletor() -> EconomicBrazilAsync:
    return EconomicBrazilAsync()


def test_fred_sem_terminal_falha_com_mensagem_clara(sem_terminal):
    """Antes o `input()` levantava EOFError no meio da coleta (exit 1)."""
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        set_fred_api_key()


def test_quandl_sem_terminal_falha_com_mensagem_clara(sem_terminal):
    with pytest.raises(RuntimeError, match="QUANDL_API_KEY"):
        set_quandl_api_key()


def test_fred_ausente_pula_a_coleta_em_vez_de_encerrar(coletor, monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr(
        "economic_brazil.coleta_dados.economic_data_brazil_async.load_dotenv",
        lambda *args, **kwargs: None,
    )

    dados = asyncio.run(coletor.dados_fred(salvar=False))

    assert "nasdaq100" not in dados.columns


def test_fonte_que_falha_nao_derruba_as_outras(coletor, monkeypatch):
    """`asyncio.gather` sem return_exceptions matava todas as fontes."""
    indice = pd.date_range("2020-01-01", periods=3, freq="MS")

    async def bcb_quebrado(**kwargs):
        raise RuntimeError("BCB fora do ar")

    async def ibge_ok(**kwargs):
        return pd.DataFrame({"ipca": [1.0, 2.0, 3.0]}, index=indice)

    monkeypatch.setattr(coletor, "dados_banco_central", bcb_quebrado)
    monkeypatch.setattr(coletor, "dados_ibge", ibge_ok)

    dados = asyncio.run(
        coletor.dados_brazil(
            dados_bcb=True,
            dados_ibge_codigos=True,
            dados_expectativas_inflacao=False,
            dados_ibge_link=False,
            dados_ipeadata=False,
            dados_google_trends=False,
            dados_fred=False,
            sem_dados_faltantes=False,
            salvar=False,
        )
    )

    assert "ipca" in dados.columns
    assert int(dados["ipca"].notna().sum()) == 3


def test_indice_duplicado_e_deduplicado_com_aviso(coletor, monkeypatch, capsys):
    """Indice duplicado nao estoura: virava coluna 100% NaN, em silencio."""
    duplicado = pd.DatetimeIndex([pd.Timestamp("2020-01-01")] * 2)

    async def ibge_duplicado(**kwargs):
        return pd.DataFrame({"ipca": [1.0, 2.0]}, index=duplicado)

    monkeypatch.setattr(coletor, "dados_ibge", ibge_duplicado)

    dados = asyncio.run(
        coletor.dados_brazil(
            dados_bcb=False,
            dados_ibge_codigos=True,
            dados_expectativas_inflacao=False,
            dados_ibge_link=False,
            dados_ipeadata=False,
            dados_google_trends=False,
            dados_fred=False,
            sem_dados_faltantes=False,
            salvar=False,
        )
    )

    assert "ipca" in dados.columns
    assert dados["ipca"].notna().sum() == 1
    assert dados["ipca"].dropna().iloc[0] == 2.0
    assert "IBGE-SIDRA" in capsys.readouterr().out
