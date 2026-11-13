"""Portoes de publicacao: bloqueia o que nao passou.

Uso:
    python -m scripts.gate            # falha so se nada for aprovado
    python -m scripts.gate --strict   # falha se qualquer variavel for bloqueada
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
ARTEFATOS = RAIZ / "artefatos"

# Abaixo disso a cobertura e ruido amostral: 12 pontos dao passos de 8 pontos
# percentuais, e a tolerancia de +-0.15 vira "acerte 10 de 12".
MIN_AMOSTRA_COBERTURA = 20
TOLERANCIA_COBERTURA = 0.15


def validar(metricas: dict) -> tuple[list[str], list[str]]:
    """Devolve (bloqueios, avisos). Aviso nao impede publicacao."""
    erros: list[str] = []
    avisos: list[str] = []
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

    n_avaliacao = metricas.get("n_avaliacao_cobertura", 0)
    for nivel, cobertura in metricas["cobertura_empirica"].items():
        if cobertura is None:
            continue
        alvo = int(nivel.strip("%")) / 100

        # O conformal garante cobertura >= nominal. Intervalo largo demais e
        # conservador, nao errado: avisa, mas nao bloqueia. So a subcobertura
        # significa confianca indevida e barra a publicacao.
        if cobertura < alvo - TOLERANCIA_COBERTURA:
            mensagem = (
                f"{nome}: cobertura {nivel} = {cobertura:.2f} abaixo de "
                f"{alvo - TOLERANCIA_COBERTURA:.2f}"
            )
            if n_avaliacao >= MIN_AMOSTRA_COBERTURA:
                erros.append(mensagem)
            else:
                avisos.append(f"{mensagem} (amostra insuficiente: {n_avaliacao})")
        elif cobertura > alvo + TOLERANCIA_COBERTURA:
            avisos.append(
                f"{nome}: cobertura {nivel} = {cobertura:.2f} bem acima de "
                f"{alvo:.2f}; intervalo conservador demais"
            )

    return erros, avisos


def main(estrito: bool = False) -> int:
    arquivos = sorted(ARTEFATOS.glob("metrics-*.json"))
    if not arquivos:
        print("::error::nenhum artefato de metricas encontrado")
        return 1

    erros: list[str] = []
    avisos: list[str] = []
    linhas: list[str] = []
    aprovados: list[str] = []

    for arquivo in arquivos:
        metricas = json.loads(arquivo.read_text(encoding="utf-8"))
        bloqueios, alertas = validar(metricas)
        erros.extend(bloqueios)
        avisos.extend(alertas)
        if not bloqueios:
            aprovados.append(metricas["variavel"])
        estado = "bloqueado" if bloqueios else "ok"
        linhas.append(
            f"| {metricas['nome']} | {metricas['campeao']} | "
            f"{metricas['mase_campeao']:.2f} | {metricas['previsao']:.2f} | "
            f"{metricas['data_previsao']} | {estado} |"
        )

    ARTEFATOS.mkdir(exist_ok=True)
    (ARTEFATOS / "aprovados.json").write_text(
        json.dumps(aprovados, ensure_ascii=False), encoding="utf-8"
    )

    resumo = [
        "## Portoes de publicacao",
        "",
        "| Variavel | Campeao | MASE | Previsao | Data | Estado |",
        "|---|---|---|---|---|---|",
        *linhas,
        "",
        f"**Aprovados para publicacao:** {', '.join(aprovados) or 'nenhum'}",
        "",
        "### Bloqueios",
        *([f"- {erro}" for erro in erros] or ["- nenhum"]),
        "",
        "### Avisos",
        *([f"- {aviso}" for aviso in avisos] or ["- nenhum"]),
    ]
    texto = "\n".join(resumo)

    Path("gate_resumo.md").write_text(texto, encoding="utf-8")
    print(texto)

    if estrito and (erros or avisos):
        return 1
    return 0 if aprovados else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    sys.exit(main(estrito=parser.parse_args().strict))
