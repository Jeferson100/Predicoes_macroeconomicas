"""Gera dados/data_quality.json e falha se uma serie-alvo estiver obsoleta.

Uso: python scripts/data_quality.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]

LIMITES_DIAS = {
    "selic": 45,
    "ipca": 45,
    "pib": 120,
    "taxa_desocupacao": 60,
    "producao_industrial": 60,
}


def avaliar(df: pd.DataFrame, hoje: pd.Timestamp | None = None) -> dict:
    hoje = pd.Timestamp(date.today()) if hoje is None else pd.Timestamp(hoje)
    series: dict[str, dict] = {}
    erros: list[str] = []

    for coluna, limite in LIMITES_DIAS.items():
        if coluna not in df.columns:
            erros.append(f"{coluna}: serie ausente do arquivo")
            continue

        serie = df[coluna]
        ultima = serie.last_valid_index()
        dias = (hoje - ultima).days if ultima is not None else None
        recentes = serie.dropna().tail(3)

        series[coluna] = {
            "ultima_data": str(ultima.date()) if ultima is not None else None,
            "dias_desde_ultima": dias,
            "missing_pct": round(float(serie.isna().mean()) * 100, 2),
            "repetido_3x": bool(len(recentes) == 3 and recentes.nunique() == 1),
            "limite_dias": limite,
        }

        if dias is None or dias > limite:
            erros.append(f"{coluna}: {dias} dias sem observacao (limite {limite})")

    return {"gerado_em": str(hoje.date()), "series": series, "erros": erros}


def main(caminho: Path | None = None) -> int:
    caminho = caminho or RAIZ / "dados" / "economic_data_brazil.pkl"
    if not caminho.exists():
        print(f"::error::arquivo de dados nao encontrado: {caminho}")
        return 1

    relatorio = avaliar(pd.read_pickle(caminho))
    destino = caminho.parent / "data_quality.json"
    destino.write_text(json.dumps(relatorio, indent=2, ensure_ascii=False), encoding="utf-8")

    for erro in relatorio["erros"]:
        print(f"::error::{erro}")

    print(json.dumps(relatorio["series"], indent=2, ensure_ascii=False))
    return 1 if relatorio["erros"] else 0


if __name__ == "__main__":
    sys.exit(main())
