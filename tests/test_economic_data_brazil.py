from economic_brazil.economic_data_brazil import data_economic
import pandas as pd


def test_no_missing_data():
    df = data_economic()
    assert not df.isna().any().any()


# write a test for check the index is datetime
def test_index_is_datetime():
    df = data_economic()
    assert isinstance(df.index, pd.DatetimeIndex)


# write a test for check the columns
def test_columns():
    df = data_economic()
    colunas_possiveis = [
        "selic",
        "IPCA-EX2",
        "IPCA-EX3",
        "IPCA-MS",
        "IPCA-MA",
        "IPCA-EX0",
        "IPCA-EX1",
        "IPCA-DP",
        "expectativas_inflacao",
        "meta_inflacao",
        "inflacao_efetiva",
        "diferenca_meta_efetiva",
        "ipca",
        "pib",
        "despesas_publica",
        "capital_fixo",
        "producao_industrial_manufatureira",
    ]
    assert all(column in colunas_possiveis for column in df.columns)
