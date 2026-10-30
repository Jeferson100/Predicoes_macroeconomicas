"""Portoes de publicacao: se falhar, nada e publicado.

Uso: python -m scripts.gate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARTEFATOS = RAIZ / "artefatos"


def validar(metricas: dict) -> list[str]:
    erros: list[str] = []
    nome = metricas["variavel"]

    if metricas["n_observacoes"] < 60:
        erros.append(f"{nome}: apenas {metricas['n_observacoes']} observacoes")

    if metricas["mase_campeao"] >= 1.0 and metricas["campeao"] != "random_walk":
        erros.append(
            f"{nome}: campeao {metricas['campeao']} com MASE " f"{metricas['mase_campeao']} >= 1.0"
        )

    limite_inferior, limite_superior = metricas["limites_plausiveis"]
    if not limite_inferior <= metricas["previsao"] <= limite_superior:
        erros.append(
            f"{nome}: previsao {metricas['previsao']} fora de "
            f"[{limite_inferior}, {limite_superior}]"
        )

    if metricas["data_previsao"] <= metricas["ultima_observacao"]:
        erros.append(f"{nome}: data prevista nao e posterior a ultima observacao")

    for nivel, cobertura in metricas["cobertura_empirica"].items():
        if cobertura is None:
            continue
        alvo = int(nivel.strip("%")) / 100
        if not alvo - 0.15 <= cobertura <= alvo + 0.15:
            erros.append(
                f"{nome}: cobertura {nivel} = {cobertura:.2f} fora de "
                f"[{alvo - 0.15:.2f}, {alvo + 0.15:.2f}]"
            )

    return erros


def main() -> int:
    arquivos = sorted(ARTEFATOS.glob("metrics-*.json"))
    if not arquivos:
        print("::error::nenhum artefato de metricas encontrado")
        return 1

    erros: list[str] = []
    linhas: list[str] = []

    for arquivo in arquivos:
        metricas = json.loads(arquivo.read_text(encoding="utf-8"))
        erros.extend(validar(metricas))
        linhas.append(
            f"| {metricas['nome']} | {metricas['campeao']} | "
            f"{metricas['mase_campeao']:.2f} | {metricas['previsao']:.2f} | "
            f"{metricas['data_previsao']} |"
        )

    resumo = [
        "## Portoes de publicacao",
        "",
        "| Variavel | Campeao | MASE | Previsao | Data |",
        "|---|---|---|---|---|",
        *linhas,
        "",
        "### Bloqueios",
        *([f"- {erro}" for erro in erros] or ["- nenhum"]),
    ]
    texto = "\n".join(resumo)

    Path("gate_resumo.md").write_text(texto, encoding="utf-8")
    print(texto)
    return 1 if erros else 0


if __name__ == "__main__":
    sys.exit(main())
