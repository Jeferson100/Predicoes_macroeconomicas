import sys

sys.path.append("..")
import pandas as pd
import pytest
from economic_brazil.processando_dados.estacionaridade import Estacionaridade


@pytest.fixture
# pylint: disable=W0621
def sample_data() -> pd.DataFrame:
    # Dados de exemplo para os testes
    data = {"A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "B": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
    return pd.DataFrame(data)


# pylint: disable=W0621


def test_kpss_adf(sample_data: pd.DataFrame) -> None:
    est = Estacionaridade()
    result = est.test_kpss_adf(sample_data)
    assert not result.empty, "O resultado do teste KPSS e ADF não deve estar vazio"
    assert all(
        result.columns
        == [
            "Teste_kpss",
            "Estacionaria_ou_nâo_kpss",
            "Teste_adf",
            "Estacionaria_ou_nâo_adf",
        ]
    ), "As colunas do resultado estão incorretas"


def test_report_ndiffs(sample_data: pd.DataFrame) -> None:
    est = Estacionaridade()
    result = est.report_ndiffs(sample_data)
    assert not result.empty, "O resultado do report_ndiffs não deve estar vazio"
    assert (
        "Ndifis" in result.columns
    ), "A coluna 'Ndifis' deve estar presente no resultado"


def test_corrigindo_nao_estacionaridade(sample_data: pd.DataFrame) -> None:
    est = Estacionaridade()
    result = est.corrigindo_nao_estacionaridade(sample_data, "A")
    assert (
        not result.empty
    ), "O resultado de corrigindo_nao_estacionaridade não deve estar vazio"
    assert all(
        result.columns == sample_data.columns
    ), "As colunas do resultado devem ser as mesmas que as da entrada"
