"""Intervalos de previsao por conformal split temporal.

Diferente do `MapieRegressor(cv=5)` usado hoje, a calibracao usa residuos
out-of-fold de origem movel (os mesmos do backtest), nunca KFold — que em
serie temporal treina com o futuro e calibra com o passado.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

from economic_brazil.forecasting.backtest import erros_absolutos


def quantil_conformal(erros, alpha: float) -> float:
    """Quantil (1 - alpha) dos erros absolutos de calibracao."""
    erros = np.asarray(erros, dtype=float).ravel()
    if erros.size == 0:
        raise ValueError("sem residuos para calibrar")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha fora de (0, 1): {alpha}")
    return float(np.quantile(erros, 1 - alpha, method="higher"))


def dividir_calibracao(backtest, n_calib: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """Separa os erros em (avaliacao, calibracao) preservando a ordem temporal."""
    return _separar(erros_absolutos(backtest), n_calib)


def _separar(erros, n_calib: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fatia a calibracao no maximo na metade, para sobrar avaliacao.

    O n_calib efetivo e recalculado aqui e devolvido por `_n_calib_efetivo`,
    porque o chamador nao pode assumir que o pedido foi atendido.
    """
    erros = np.asarray(erros, dtype=float).ravel()
    n_calib = _n_calib_efetivo(len(erros), n_calib)
    return erros[:-n_calib], erros[-n_calib:]


def _n_calib_efetivo(n_erros: int, n_calib: int) -> int:
    return max(1, min(n_calib, n_erros // 2))


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
    """Cobertura do intervalo calibrado, medida fora da janela de calibracao."""
    erros = erros_absolutos(backtest)
    avaliacao, calibracao = _separar(erros, n_calib)
    if avaliacao.size == 0:
        return None
    q = quantil_conformal(calibracao, alpha)
    return float(np.mean(avaliacao <= q))


def amplitude_media(intervalo: Tuple[float, float]) -> float:
    return float(intervalo[1] - intervalo[0])
