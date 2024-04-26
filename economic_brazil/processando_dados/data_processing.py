import sys
sys.path.append("..")
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import warnings

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


def backcasting_nan(dados):
    """
    Fills missing values in the input DataFrame with interpolated values using the XGBoost regressor model.

    Parameters:
        dados (pd.DataFrame): The input DataFrame containing the data to be processed.

    Returns:
        pd.DataFrame: The processed DataFrame with missing values filled in.
    """
    colunas = dados.columns.tolist()
    np.random.seed(0)
    dados_bac = dados.copy()

    for i in colunas:
        dados_interpolacao = dados.loc[: dados[i].last_valid_index()]

        treino_interpolacao = dados_interpolacao.dropna(subset=[i])
        predicao_interpolacao = dados_interpolacao[dados_interpolacao[i].isnull()]

        Y_treino = treino_interpolacao[i]
        treino_interpolacao = treino_interpolacao.drop(i, axis=1)
        predicao_interpolacao = predicao_interpolacao.drop(i, axis=1)

        predicao_interpolacao = predicao_interpolacao.dropna(
            thresh=len(predicao_interpolacao) - 4, axis=1
        )
        Y_treino = Y_treino[Y_treino.index.isin(treino_interpolacao.index)]
        treino_interpolacao = treino_interpolacao.loc[:, predicao_interpolacao.columns]

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("model", XGBRegressor(random_state=0)),
            ]
        )

        try:
            pipeline.fit(treino_interpolacao, Y_treino)
            predicao = pipeline.predict(predicao_interpolacao)
            interpolados = pd.DataFrame(
                predicao, index=predicao_interpolacao.index, columns=[i]
            )
            dados_bac.update(interpolados)
        except Exception as e:
            print(f"Error processing column {i}: {str(e)}")

    return dados_bac

def criando_defasagens(base,numero_defasagens=4):
    base_def = base.copy()
    for j in range(numero_defasagens):
        for i in base.columns:
            base_def[i+str('_lags_')+str(j+1)] = base_def[i].shift(j+1)
    return base_def




