import sys

sys.path.append("..")
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
from pmdarima import arima
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from typing import Optional, List, Any

warnings.filterwarnings("ignore")


def criando_dummy_covid(dados, inicio_periodo: str, fim_periodo: str) -> pd.DataFrame:
    """
    Cria uma coluna 'dummy_covid' no DataFrame 'dados' com valores 1 para os registros dentro do período de datas especificado e 0 para os de fora.

    Args:
        dados (pd.DataFrame): O DataFrame contendo os dados.
        inicio_periodo (str): A data de início do período em formato 'YYYY-MM-DD'.
        fim_periodo (str): A data de fim do período em formato 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: O DataFrame 'dados' com a coluna 'dummy_covid' adicionada.
    """
    periodo = pd.date_range(start=inicio_periodo, end=fim_periodo)
    dados["dummy_covid"] = np.where(dados.index.isin(periodo), 1, 0)
    return dados


def backcasting_nan(dados: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values in the input DataFrame with interpolated values using the XGBoost regressor model.

    Parameters:
        dados (pd.DataFrame): The input DataFrame containing the data to be processed.

    Returns:
        pd.DataFrame: The processed DataFrame with missing values filled in.
    """
    np.random.seed(0)
    dados_bac = dados.copy()
    colunas = dados.columns.tolist()
    for i in colunas:
        dados_interpolacao = dados.copy()

        dados_interpolacao = dados_interpolacao.loc[
            : dados_interpolacao[i].last_valid_index()
        ]

        # Separate values not NaN and NaN in two different dataframes
        treino_interpolacao = dados_interpolacao[dados_interpolacao[i].notnull()]
        predicao_interpolacao = dados_interpolacao[dados_interpolacao[i].isnull()]

        # Create Y_treino column and remove from treino_interpolacao
        Y_treino = treino_interpolacao[i]
        treino_interpolacao = treino_interpolacao.drop(i, axis=1)

        predicao_interpolacao = predicao_interpolacao.drop(i, axis=1)

        # Remove columns with more than 10 NaN values
        predicao_interpolacao = predicao_interpolacao.dropna(
            thresh=len(predicao_interpolacao) - 4, axis=1
        )

        Y_treino = Y_treino[Y_treino.index.isin(treino_interpolacao.index)]

        treino_interpolacao = treino_interpolacao.loc[
            :, predicao_interpolacao.columns.tolist()
        ]

        # Fill NaN values in remaining columns with the mean
        treino_interpolacao = treino_interpolacao.fillna(treino_interpolacao.mean())
        predicao_interpolacao = predicao_interpolacao.fillna(
            predicao_interpolacao.mean()
        )
        # Filter prediction dataframe to include only columns present in training dataframe
        # Create the model
        model = XGBRegressor(random_state=0)

        # Train the model
        model.fit(treino_interpolacao.values, Y_treino.to_numpy().reshape(-1, 1))
        predicao = model.predict(predicao_interpolacao.values)
        interpolados = pd.DataFrame(predicao, index=predicao_interpolacao.index)
        try:
            interpolados.columns = [i]
            dados_bac.fillna(interpolados, inplace=True)
        except ValueError:
            dados_bac[i] = dados[i]

    return dados_bac


def criando_defasagens(base: pd.DataFrame, numero_defasagens: int = 4) -> pd.DataFrame:
    base_def = base.copy()
    for j in range(numero_defasagens):
        for i in base.columns:
            col_name = f"{i}_lags_{j+1}"
            base_def[col_name] = base_def[i].shift(j + 1)
    return base_def


def corrigindo_nan_arima(
    dados_est,
    coluna: str,
    test_arima: str = "adf",
    max_p_arima: int = 3,
    max_q_arima: int = 3,
    d_arima: Optional[int] = None,
    seasonal_arima: bool = False,
    start_p_arima: int = 0,
    D_arima: int = 0,
    trace_arima: bool = True,
    error_action_arima: str = "ignore",
    suppress_warnings_arima: bool = True,
    stepwise_arima: bool = True,
) -> pd.DataFrame:
    dados_sem = dados_est.loc[:, dados_est.columns != coluna].copy()
    for k in dados_est.loc[:, dados_est.columns != coluna].columns[
        dados_sem.isnull().sum() > 0
    ]:
        model_arima = arima.auto_arima(
            dados_sem[k].dropna().values,
            start_p=1,
            start_q=1,
            test=test_arima,  # use adftest to find optimal 'd'
            max_p=max_p_arima,
            max_q=max_q_arima,  # maximum p and q
            m=1,  # frequency of series
            d=d_arima,  # let model determine 'd'
            seasonal=seasonal_arima,  # No Seasonality
            start_P=start_p_arima,
            D=D_arima,
            trace=trace_arima,
            error_action=error_action_arima,
            suppress_warnings=suppress_warnings_arima,
            stepwise=stepwise_arima,
        )
        model_arima.fit(dados_sem[k].values)
        a = model_arima.predict(n_periods=int(dados_sem[k].isnull().sum()))
        for i in a:
            dados_sem[k] = dados_sem[k].fillna(i, limit=1)
    return dados_sem


def criando_mes_ano_dia(
    dados: pd.DataFrame,
    mes: bool = False,
    ano: bool = False,
    dia: bool = False,
    dummy: bool = False,
    trimestre: bool = False,
    coluns: Optional[List[str]] = None,
):
    if mes:
        dados["mes"] = pd.to_datetime(dados.index).month  # type: ignore
    if ano:
        dados["ano"] = pd.to_datetime(dados.index).year  # type: ignore
    if dia:
        dados["dia"] = pd.to_datetime(dados.index).day  # type: ignore
    if trimestre:
        dados["trimestre"] = pd.to_datetime(dados.index).quarter  # type: ignore
    if dummy:
        if coluns is None:
            coluns = ["mes", "ano", "dia", "trimestre"]
        dados_dia_mes_ano = pd.get_dummies(
            dados,
            columns=coluns,
            prefix=[i for i in coluns],
            prefix_sep="_",
            drop_first=True,
        ).astype(float)
        dados = dados_dia_mes_ano.copy()

    dados = dados.loc[:, ~dados.columns.duplicated()]
    return dados


def escalando_dados(dados: np.ndarray, tipo: str = "minmax") -> tuple[np.ndarray, Any]:
    if tipo == "minmax":
        scaler = MinMaxScaler()
        scaler.fit(dados)
        dados = scaler.transform(dados)
    else:
        scaler = StandardScaler()
        scaler.fit(dados)
        dados = scaler.transform(dados)
    return dados, scaler
