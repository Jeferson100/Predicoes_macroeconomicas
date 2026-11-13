"""Intervalos de previsao por conformal split temporal.

Diferente do `MapieRegressor(cv=5)` usado hoje, a calibracao usa residuos
out-of-fold de origem movel (os mesmos do backtest), nunca KFold — que em
serie temporal treina com o futuro e calibra com o passado.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from economic_brazil.forecasting.backtest import erros_absolutos

MIN_PONTOS_AVALIACAO = 20


def quantil_conformal(erros, alpha: float) -> float:
    """Quantil conformal dos erros absolutos de calibracao.

    A correcao `ceil((n + 1)(1 - alpha)) / n` e o que garante a validade de
    amostra finita. Sem ela, com n pequeno o quantil empirico puro fica baixo
    e o intervalo sai estreito demais.
    """
    erros = np.asarray(erros, dtype=float).ravel()
    if erros.size == 0:
        raise ValueError("sem residuos para calibrar")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha fora de (0, 1): {alpha}")
    n = erros.size
    nivel = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(erros, nivel, method="higher"))


def _n_calib_efetivo(n_erros: int, n_calib: int) -> int:
    """Reserva pelo menos MIN_PONTOS_AVALIACAO para medir cobertura."""
    if n_erros <= MIN_PONTOS_AVALIACAO:
        return max(1, n_erros // 2)
    return max(1, min(n_calib, n_erros - MIN_PONTOS_AVALIACAO))


def pontos_avaliacao(backtest, n_calib: int = 24) -> int:
    """Quantos pontos de fato entram na medicao de cobertura."""
    n_erros = len(erros_absolutos(backtest))
    return max(0, n_erros - _n_calib_efetivo(n_erros, n_calib))


def intervalos(
    centro: float,
    erros_calibracao,
    alphas: Sequence[float] = (0.2, 0.05),
) -> Dict[str, Tuple[float, float]]:
    """Devolve {nivel: (lower, upper)} para cada alpha."""
    return {
        f"{round((1 - alpha) * 100)}%": (
            float(centro) - quantil_conformal(erros_calibracao, alpha),
            float(centro) + quantil_conformal(erros_calibracao, alpha),
        )
        for alpha in alphas
    }


def cobertura_empirica(backtest, alpha: float, n_calib: int = 24) -> float | None:
    """Cobertura rolling: em cada origem o quantil usa apenas o passado.

    A versao anterior calibrava nas ultimas origens e media a cobertura nas
    primeiras — o quantil via o futuro. Aqui cada erro `i` e comparado ao
    quantil dos `n_calib` erros imediatamente anteriores a ele.
    """
    erros = erros_absolutos(backtest)
    n_calib = _n_calib_efetivo(len(erros), n_calib)
    if len(erros) <= n_calib:
        return None
    cobertos = [
        erros[i] <= quantil_conformal(erros[i - n_calib : i], alpha)
        for i in range(n_calib, len(erros))
    ]
    return float(np.mean(cobertos))


def amplitude_media(intervalo: Tuple[float, float]) -> float:
    return float(intervalo[1] - intervalo[0])
