from economic_brazil.coleta_dados.coleta_economic_brazil import (
    dados_bcb,
    dados_ibge_link,
    dados_ibge_codigos,
    dados_expectativas_focus,
    metas_inflacao,
)

# write tests for tratando_dados_bcb
def test_tratando_dados_bcb():
    dados = dados_bcb({"selic": 4189}, "2000-01-01")
    assert len(dados) > 0


##write tests for dados_ibge_codigos
def test_dados_ibge_codigos():
    dados = dados_ibge_codigos()
    assert len(dados) > 0


##write tests for dados_ibge_link
def test_dados_ibge_link():
    dados = dados_ibge_link()
    assert len(dados) > 0


##write tests for metas_inflacao
def test_metas_inflacao():
    dados = metas_inflacao()
    assert len(dados) > 0


##write tests for dados_expectativas_focus
def test_dados_expectativas_focus():
    dados = dados_expectativas_focus()
    assert len(dados) > 0
