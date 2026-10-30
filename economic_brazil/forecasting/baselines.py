"""Baselines obrigatorios.

Nenhum modelo e publicado se nao bater estes em MASE. Hoje (artefatos de
out/26) SELIC, PIB e desocupacao perdem para o random walk.
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def random_walk(y, h: int) -> np.ndarray:
    """Repete o ultimo valor."""
    y = np.asarray(y, dtype=float)
    return np.repeat(y[-1], h)


def seasonal_naive(y, h: int, season: int = 12) -> np.ndarray:
    """Repete o valor de `season` periodos atras."""
    y = np.asarray(y, dtype=float)
    if season <= 0 or len(y) < season:
        return random_walk(y, h)
    indices = (np.arange(len(y), len(y) + h) - season) % len(y)
    return y[indices]


def media(y, h: int) -> np.ndarray:
    """Media historica."""
    y = np.asarray(y, dtype=float)
    return np.repeat(float(np.mean(y)), h)


def drift(y, h: int) -> np.ndarray:
    """Random walk com deriva (inclinacao media do periodo)."""
    y = np.asarray(y, dtype=float)
    if len(y) < 2:
        return random_walk(y, h)
    inclinacao = (y[-1] - y[0]) / (len(y) - 1)
    return y[-1] + inclinacao * np.arange(1, h + 1)


BASELINES: Dict[str, Callable[..., np.ndarray]] = {
    "random_walk": random_walk,
    "seasonal_naive": seasonal_naive,
    "media": media,
    "drift": drift,
}
