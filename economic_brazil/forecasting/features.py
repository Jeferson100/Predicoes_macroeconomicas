"""Construcao de features sem vazamento.

Todas as colunas sao defasagens de valores passados ou funcoes deterministicas
do calendario. Nada e ajustado aqui, entao treino, teste e futuro passam pelo
mesmo codigo por construcao.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def periodo_sazonal(indice: pd.Index) -> int:
    """12 para series mensais, 4 para trimestrais, 0 quando nao ha sazonalidade."""
    if not isinstance(indice, pd.DatetimeIndex) or len(indice) < 2:
        return 0
    frequencia = (pd.infer_freq(indice) or "").upper()
    if frequencia.startswith("M"):
        return 12
    if frequencia.startswith("Q"):
        return 4
    return 0


def _sazonalidade(indice: pd.DatetimeIndex, periodo: int) -> Optional[pd.DataFrame]:
    if periodo == 12:
        posicao = indice.month - 1
    elif periodo == 4:
        posicao = indice.quarter - 1
    else:
        return None
    angulo = 2 * np.pi * posicao / periodo
    return pd.DataFrame(
        {"sazonal_sen": np.sin(angulo), "sazonal_cos": np.cos(angulo)}, index=indice
    )


def _colunas_defasadas(
    serie: pd.Series,
    lags: Sequence[int],
    exogenas: Optional[pd.DataFrame],
    defasagens_exogenas: Sequence[int],
) -> pd.DataFrame:
    X = pd.DataFrame(index=serie.index)
    for lag in lags:
        X[f"alvo_lag_{lag}"] = serie.shift(lag)
    if exogenas is not None:
        exogenas = exogenas.reindex(serie.index)
        for coluna in exogenas.columns:
            for lag in defasagens_exogenas:
                X[f"{coluna}_lag_{lag}"] = exogenas[coluna].shift(lag)
    return X


def montar_dataset(
    serie: pd.Series,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    exogenas: Optional[pd.DataFrame] = None,
    defasagens_exogenas: Sequence[int] = (1, 2),
    sazonalidade: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Devolve (X, y) alinhados e sem NaN."""
    serie = pd.Series(serie).astype(float).sort_index()
    serie = serie[~serie.index.duplicated(keep="last")]

    X = _colunas_defasadas(serie, lags, exogenas, defasagens_exogenas)
    periodo = periodo_sazonal(serie.index) if sazonalidade else 0
    sazonal = _sazonalidade(serie.index, periodo)
    if sazonal is not None:
        X = X.join(sazonal)

    dados = pd.concat([serie.rename("alvo"), X], axis=1).dropna()
    return dados.drop(columns="alvo"), dados["alvo"]


def montar_linha_futura(
    serie: pd.Series,
    lags: Sequence[int] = (1, 2, 3, 6, 12),
    exogenas: Optional[pd.DataFrame] = None,
    defasagens_exogenas: Sequence[int] = (1, 2),
    sazonalidade: bool = True,
    periodo_offset: int = 1,
) -> pd.DataFrame:
    """Features para prever o proximo periodo a partir do ultimo valor observado.

    Usa apenas `serie.iloc[-lag]` e `exogenas.iloc[-lag]`, entao nao precisa
    de exogenas futuras nem de preenchimento artificial.
    """
    serie = pd.Series(serie).astype(float).sort_index()
    serie = serie[~serie.index.duplicated(keep="last")].dropna()

    valores = serie.to_numpy()
    linha: dict[str, float] = {}
    for lag in lags:
        linha[f"alvo_lag_{lag}"] = float(valores[-lag]) if lag <= len(valores) else np.nan

    if exogenas is not None:
        exogenas = exogenas.reindex(serie.index).dropna(how="all")
        for coluna in exogenas.columns:
            for lag in defasagens_exogenas:
                valores_exo = exogenas[coluna].to_numpy()
                linha[f"{coluna}_lag_{lag}"] = (
                    float(valores_exo[-lag]) if lag <= len(valores_exo) else np.nan
                )

    periodo = periodo_sazonal(serie.index) if sazonalidade else 0
    meses = 3 if periodo == 4 else 1
    futuro = serie.index[-1] + pd.DateOffset(months=meses * periodo_offset)

    if periodo:
        sazonal = _sazonalidade(pd.DatetimeIndex([futuro]), periodo)
        if sazonal is not None:
            for coluna in sazonal.columns:
                linha[coluna] = float(sazonal[coluna].iloc[0])

    indice = pd.DatetimeIndex([futuro])

    return pd.DataFrame([linha], index=indice)
