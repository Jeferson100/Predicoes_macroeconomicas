"""Alvo ancorado.

Arvores e redes nao extrapolam o intervalo visto no treino: o SELIC previsto
nunca sai de [min, max] historico. Pior, em 4 de 5 variaveis o modelo atual
perde para "repetir o ultimo valor". Modelar `y - ancora` resolve os dois:
o modelo aprende a correcao, e a ancora garante que ele so ganha se somar
sinal de verdade.
"""

from __future__ import annotations

import numpy as np


class AlvoAncorado:
    """Treina em `y - ancora(y)` e reconstroi o nivel na previsao."""

    def __init__(self, modelo=None, ancora: str = "random_walk") -> None:
        self.modelo = modelo
        self.ancora = ancora
        self.ancora_treino_: float | None = None

    def _ancora(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=float).ravel()
        if self.ancora == "media":
            return float(np.mean(y))
        if self.ancora == "random_walk":
            return float(y[-1])
        raise ValueError(f"ancora desconhecida: {self.ancora!r}")

    def fit(self, X, y, **kwargs) -> "AlvoAncorado":
        y = np.asarray(y, dtype=float).ravel()
        self.ancora_treino_ = self._ancora(y)
        self.modelo.fit(X, y - self.ancora_treino_, **kwargs)
        return self

    def predict(self, X, ancora_atual: float | None = None) -> np.ndarray:
        if self.ancora_treino_ is None:
            raise RuntimeError("AlvoAncorado.predict chamado antes de fit")
        base = self.ancora_treino_ if ancora_atual is None else float(ancora_atual)
        previsao = np.asarray(self.modelo.predict(X), dtype=float).ravel()
        return previsao + base

    def get_params(self, deep: bool = True) -> dict:  # pylint: disable=unused-argument
        return {"modelo": self.modelo, "ancora": self.ancora}

    def set_params(self, **params) -> "AlvoAncorado":
        for chave, valor in params.items():
            setattr(self, chave, valor)
        return self
