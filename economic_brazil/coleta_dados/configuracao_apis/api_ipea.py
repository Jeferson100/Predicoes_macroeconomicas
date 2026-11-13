"""Cliente minimo da API OData do IPEADATA.

O `ipeadatapy` abre uma conexao nova a cada chamada e faz duas requisicoes
por serie (metadados + valores). O servidor leva ~22s no primeiro handshake
e ~0,2s por requisicao depois disso, entao 25 series passavam de 600s e o
job de coleta cancelava a fonte inteira.

Aqui a sessao e unica e reaproveitada (o pool do urllib3 e thread-safe) e so
os valores sao buscados: o handshake e pago uma vez, nao uma vez por serie.
"""

from __future__ import annotations

import time
from functools import lru_cache

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

BASE = "https://www.ipeadata.gov.br/api/odata4"
TIMEOUT_SEGUNDOS = 60
TENTATIVAS = 3


@lru_cache(maxsize=1)
def _sessao() -> requests.Session:
    """Sessao unica por processo, com pool de conexoes do urllib3.

    `lru_cache` ja serializa a criacao entre threads.
    """
    sessao = requests.Session()
    sessao.mount(
        "https://",
        HTTPAdapter(pool_connections=4, pool_maxsize=8),
    )
    return sessao


def _url(codigo: str) -> str:
    return f"{BASE}/ValoresSerie(SERCODIGO='{codigo}')"


def _parse(valores: list[dict], codigo: str) -> pd.Series:
    if not valores:
        raise ValueError(f"serie vazia: {codigo}")

    dados = pd.DataFrame(valores)
    # VALDATA vem como 2000-01-01T00:00:00-02:00. Cortar a string evita que
    # o fuso desloque o dia.
    dados["VALDATA"] = pd.to_datetime(dados["VALDATA"].str.slice(0, 10))
    serie = dados.set_index("VALDATA")["VALVALOR"].astype(float).sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]
    serie.name = codigo
    return serie


def buscar_serie(codigo: str) -> pd.Series:
    """Busca uma serie no IPEADATA, com retry e sessao reaproveitada."""
    ultimo_erro: Exception | None = None

    for tentativa in range(1, TENTATIVAS + 1):
        try:
            resposta = _sessao().get(_url(codigo), timeout=TIMEOUT_SEGUNDOS)
            resposta.raise_for_status()
            return _parse(resposta.json().get("value", []), codigo)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            ultimo_erro = exc
            if tentativa < TENTATIVAS:
                time.sleep(2**tentativa)

    raise RuntimeError(f"falha ao buscar a serie {codigo}") from ultimo_erro
