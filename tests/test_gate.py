"""Portoes de publicacao: bloqueio so com amostra suficiente."""

import json

from scripts import gate


def _metricas(**extra):
    base = {
        "variavel": "selic",
        "nome": "selic",
        "n_observacoes": 274,
        "mase_campeao": 0.66,
        "campeao": "ridge",
        "limites_plausiveis": [-5.0, 30.0],
        "previsao": 13.75,
        "data_previsao": "2026-11-01",
        "ultima_observacao": "2026-10-01",
        "cobertura_empirica": {"80%": 0.8, "95%": 0.95},
        "n_avaliacao_cobertura": 24,
    }
    base.update(extra)
    return base


def test_metricas_boas_nao_geram_bloqueio():
    erros, avisos = gate.validar(_metricas())

    assert erros == []
    assert avisos == []


def test_mase_acima_de_um_bloqueia():
    erros, _ = gate.validar(_metricas(mase_campeao=1.09, campeao="seasonal_naive"))

    assert any("MASE" in erro for erro in erros)


def test_cobertura_ruim_com_amostra_suficiente_bloqueia():
    erros, avisos = gate.validar(_metricas(cobertura_empirica={"80%": 0.50}))

    assert any("cobertura" in erro for erro in erros)
    assert avisos == []


def test_cobertura_alta_avisa_mas_nao_bloqueia():
    """Conformal garante cobertura >= nominal: intervalo largo e conservador."""
    erros, avisos = gate.validar(_metricas(cobertura_empirica={"80%": 1.00}))

    assert erros == []
    assert any("conservador" in aviso for aviso in avisos)


def test_cobertura_ruim_com_amostra_pequena_vira_aviso():
    """Com 8 pontos a cobertura e ruido: nao pode bloquear publicacao."""
    erros, avisos = gate.validar(
        _metricas(cobertura_empirica={"80%": 0.50}, n_avaliacao_cobertura=8)
    )

    assert erros == []
    assert any("amostra insuficiente" in aviso for aviso in avisos)


def test_main_grava_so_os_aprovados(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ARTEFATOS", tmp_path)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "metrics-selic.json").write_text(json.dumps(_metricas()), encoding="utf-8")
    (tmp_path / "metrics-ipca.json").write_text(
        json.dumps(
            _metricas(variavel="ipca", nome="ipca", mase_campeao=1.09, campeao="seasonal_naive")
        ),
        encoding="utf-8",
    )

    assert gate.main() == 0
    aprovados = json.loads((tmp_path / "aprovados.json").read_text(encoding="utf-8"))
    assert aprovados == ["selic"]


def test_main_falha_quando_nada_e_aprovado(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ARTEFATOS", tmp_path)
    monkeypatch.chdir(tmp_path)

    (tmp_path / "metrics-ipca.json").write_text(
        json.dumps(
            _metricas(variavel="ipca", nome="ipca", mase_campeao=1.09, campeao="seasonal_naive")
        ),
        encoding="utf-8",
    )

    assert gate.main() == 1
    assert json.loads((tmp_path / "aprovados.json").read_text(encoding="utf-8")) == []
