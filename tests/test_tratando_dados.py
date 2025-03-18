import sys

sys.path.append("..")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from economic_brazil.processando_dados.tratando_dados import TratandoDados
from economic_brazil.coleta_dados.economic_data_brazil import EconomicBrazil
import pytest
import pandas as pd


@pytest.fixture
# pylint: disable=W0621
def sample_data() -> pd.DataFrame:
    """
    Fixture that returns a sample dataset.

    This fixture retrieves a sample dataset by calling the `data_economic` function with the `data_inicio` parameter set to '2000-01-01'. The retrieved dataset is then returned by the fixture.

    Returns:
        pandas.DataFrame: The sample dataset.
    """
    # pylint: disable=W0621
    economic_brazil = EconomicBrazil(data_inicio="2000-01-01")
    dados = economic_brazil.dados_brazil(
        dados_ibge_link=False,
        dados_expectativas_inflacao=False,
    )
    # pylint: disable=W0621
    return dados


# pylint: disable=W0621


def test_data_divisao_treino_teste(sample_data: pd.DataFrame) -> None:
    processer = TratandoDados(sample_data, data_divisao="2020-05-31")
    assert processer.data_divisao_treino_teste() == "2020-05-31"


def test_tratando_divisao_y_treino_y_teste_x_treino_x_teste(
    sample_data: pd.DataFrame,
) -> None:
    coluna_label = sample_data.columns[0]
    processer = TratandoDados(sample_data, coluna_label=coluna_label)
    # pylint: disable=W0632
    x_treino, y_treino, x_teste, y_teste = processer.tratando_divisao(
        sample_data, treino_teste=False
    )
    # pylint: disable=W0632
    assert len(x_treino) > 0
    assert len(y_teste) > 0
    assert len(x_teste) > 0
    assert len(y_treino) > 0


def test_tratando_divisao_treino_teste(sample_data: pd.DataFrame) -> None:
    coluna_label = sample_data.columns[0]
    processer = TratandoDados(sample_data, coluna_label=coluna_label)
    treino, teste = processer.tratando_divisao(sample_data)
    assert len(treino) > 0
    assert len(teste) > 0


def test_tratando_covid(sample_data: pd.DataFrame) -> None:
    processer = TratandoDados(sample_data)
    dados_covid = processer.tratando_covid(sample_data)
    assert "dummy_covid" in dados_covid.columns


def test_tratando_estacionaridade(sample_data: pd.DataFrame) -> None:
    processer = TratandoDados(sample_data)
    dados_est = processer.tratando_estacionaridade(sample_data)
    assert "selic" in dados_est.columns


def test_tratando_datas(sample_data: pd.DataFrame) -> None:
    data_processor = TratandoDados(sample_data, data_divisao="2020-05-31")
    dados_datas = data_processor.tratando_datas(sample_data)
    assert len(dados_datas.columns) > len(sample_data.columns)


def test_tratando_defasagens(sample_data: pd.DataFrame) -> None:
    data_processor = TratandoDados(
        sample_data, data_divisao="2020-05-31", numero_defasagens=4
    )
    dados_defas = data_processor.tratando_defasagens(sample_data)
    assert len(dados_defas.columns) > len(sample_data.columns)


def test_tratando_divisao_x_y(sample_data: pd.DataFrame) -> None:
    data_processor = TratandoDados(sample_data, data_divisao="2020-05-31")
    coluna_label = sample_data.columns[0]
    x, y = data_processor.tratando_divisao_x_y(sample_data, label=coluna_label)
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == sample_data.shape[1] - 1


def test_tratando_scaler(sample_data: pd.DataFrame) -> None:
    data_processor = TratandoDados(sample_data, data_divisao="2020-05-31")
    dados_scaler, scaler = data_processor.tratando_scaler(sample_data)
    assert dados_scaler.shape == sample_data.shape
    assert isinstance(scaler, (StandardScaler, MinMaxScaler))


def test_tratando_pca(sample_data: pd.DataFrame) -> None:
    data_processor = TratandoDados(sample_data, data_divisao="2020-05-31")
    pca, dados_pca = data_processor.tratando_pca(sample_data)
    assert dados_pca.shape[1] <= sample_data.shape[1]
    assert isinstance(pca, PCA)


def test_tratando_dados(sample_data: pd.DataFrame) -> None:
    coluna_label = sample_data.columns[0]
    data_processor = TratandoDados(
        sample_data,
        coluna_label=coluna_label,
        rfe=True,
        variancia=True,
        smart_correlation=True,
    )
    (
        x_treino,
        x_teste,
        y_treino,
        y_teste,
        pca_modelo,
        scaler_modelo,
        rfe_modelo,
        variancia_modelo,
        smart_correlation_modelo,
    ) = data_processor.tratando_dados()
    assert x_treino is not None
    assert x_teste is not None
    assert y_treino is not None
    assert y_teste is not None
    assert pca_modelo is not None
    assert rfe_modelo is not None
    assert variancia_modelo is not None
    assert smart_correlation_modelo is not None
    assert scaler_modelo is not None
    assert isinstance(scaler_modelo, (MinMaxScaler, StandardScaler))
    assert isinstance(pca_modelo, PCA)
