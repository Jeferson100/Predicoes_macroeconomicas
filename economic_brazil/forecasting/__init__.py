"""Previsao macro com baselines obrigatorios, backtest de origem movel e portoes.

Este pacote e aditivo: o pipeline antigo em ``codigos_rodando/`` continua
funcionando. A ideia e que cada variavel seja treinada por ``scripts/train.py``
a partir de um YAML em ``configs/``.
"""

from economic_brazil.forecasting.backtest import (
    backtest_baseline,
    backtest_supervisionado,
    mase,
    resumo,
)
from economic_brazil.forecasting.baselines import BASELINES
from economic_brazil.forecasting.features import montar_dataset
from economic_brazil.forecasting.target import AlvoAncorado

__all__ = [
    "BASELINES",
    "AlvoAncorado",
    "backtest_baseline",
    "backtest_supervisionado",
    "mase",
    "montar_dataset",
    "resumo",
]
