import sys

sys.path.append("..")
from economic_brazil.processando_dados.data_processing import (
    criando_dummy_covid,
    criando_defasagens,
    criando_mes_ano_dia,
    escalando_dados,
)
from economic_brazil.processando_dados.estacionaridade import Estacionaridade
from sklearn.decomposition import PCA
from economic_brazil.processando_dados.divisao_treino_teste import treino_test_dados
from sklearn.feature_selection import RFE, VarianceThreshold
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Any
import scipy.sparse


class TratandoDados:
    def __init__(
        self,
        df: pd.DataFrame,
        data_divisao: Optional[str] = None,
        coluna_label: Optional[str] = None,
        numero_defasagens: Optional[int] = None,
        n_components: Optional[int] = None,
        n_features_to_select: Optional[int] = None,
        variancia_threshold: Optional[float] = None,
        smart_correlated_threshold: Optional[float] = None,
        covid: Optional[bool] = True,
        estacionaridade: Optional[bool] = True,
        datas: Optional[bool] = True,
        estacionaridade_log: bool = False,
        defasagens: bool = True,
        pca: bool = True,
        scaler: bool = True,
        rfe: bool = False,
        variancia: bool = False,
        smart_correlation: bool = False,
    ) -> None:
        self.df = df
        self.scaler_modelo = None
        self.pca_modelo = None
        self.rfe_modelo = None
        self.variancia_modelo = None
        self.smart_correlation_modelo = None
        self.data_divisao = data_divisao
        self.coluna_label = coluna_label
        self.numero_defasagens = numero_defasagens
        self.n_components = n_components
        self.n_features_to_select = n_features_to_select
        self.variancia_threshold = variancia_threshold
        self.smart_correlated_threshold = smart_correlated_threshold
        self.covid = covid
        self.estacionaridade = estacionaridade
        self.datas = datas
        self.defasagens = defasagens
        self.pca = pca
        self.scaler = scaler
        self.rfe = rfe
        self.variancia = variancia
        self.smart_correlation = smart_correlation
        self.estacionaridade_log = estacionaridade_log

    def data_divisao_treino_teste(self) -> str:
        if self.data_divisao is None:
            data_inicio = self.df[-50:].index[1].strftime("%Y-%m-%d")
            print("Data divisao de treino e teste:", data_inicio)
            return data_inicio
        else:
            print("Data divisao de treino e teste:", self.data_divisao)
            return self.data_divisao

    # pylint: disable=W0632
    def tratando_divisao(
        self,
        dados: pd.DataFrame,
        treino_teste: bool = True,
        divisao_treino_teste: Optional[str] = None,
    ) -> Any:
        """
        Divide os dados em conjuntos de treino e teste.

        """
        if divisao_treino_teste is None:
            divisao_treino_teste = self.data_divisao_treino_teste()
        if self.coluna_label is None:
            self.coluna_label = "selic"

        if treino_teste:
            # pylint: disable=W0632
            treino, teste = treino_test_dados(
                dados, data_divisao=divisao_treino_teste, treino_teste=True
            )

            return treino, teste
        else:
            # pylint: disable=W0632
            x_treino, y_treino, x_teste, y_teste = treino_test_dados(
                dados, data_divisao=divisao_treino_teste, coluna=self.coluna_label
            )
            return x_treino, y_treino, x_teste, y_teste

    # pylint: disable=W0632
    def tratando_covid(
        self,
        dados: pd.DataFrame,
        inicio_periodo: str = "2020-04-01",
        fim_periodo: str = "2020-05-01",
    ) -> pd.DataFrame:
        """
        Adiciona variáveis dummy para o período COVID.
        """
        dados_covid = criando_dummy_covid(
            dados, inicio_periodo=inicio_periodo, fim_periodo=fim_periodo
        )
        return dados_covid

    def tratando_estacionaridade(
        self, dados: pd.DataFrame, coluna_label: str = "selic"
    ) -> pd.DataFrame:
        """
        Corrige a não-estacionaridade dos dados.
        """
        estacionaridade = Estacionaridade()
        dados_est = estacionaridade.corrigindo_nao_estacionaridade(dados, coluna_label)
        return dados_est

    def tratando_datas(
        self,
        dados: pd.DataFrame,
        mes: bool = True,
        trimestre: bool = True,
        dummy: bool = True,
        colunas: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Adiciona colunas de mês, trimestre e dummies aos dados.
        """
        if colunas is None:
            colunas = ["mes", "trimestre"]
        dados_datas = criando_mes_ano_dia(
            dados, mes=mes, trimestre=trimestre, dummy=dummy, coluns=colunas
        )
        return dados_datas

    def tratando_defasagens(self, dados: pd.DataFrame) -> pd.DataFrame:
        """
        Cria defasagens nos dados.
        """
        if self.numero_defasagens is None:
            self.numero_defasagens = 4

        dados_defas = criando_defasagens(
            dados, numero_defasagens=self.numero_defasagens
        )
        dados_defas = dados_defas[self.numero_defasagens :]
        if dados_defas.isnull().values.any():
            dados_defas = dados_defas.ffill()
            dados_defas = dados_defas.bfill()
        return dados_defas

    def tratando_divisao_x_y(
        self, dados: pd.DataFrame, label: str = "selic"
    ) -> tuple[np.ndarray, Any]:
        """
        Separa as variáveis independentes e dependentes.
        """
        y = dados[label].values
        x = dados.loc[:, dados.columns != label].values
        return x, y

    def tratando_scaler(
        self, dados: Union[pd.DataFrame, np.ndarray], tipo: str = "scaler"
    ) -> tuple[np.ndarray, StandardScaler]:
        """
        Escala os dados usando o método especificado.
        """
        if isinstance(dados, pd.DataFrame):
            dados_numpy = dados.values
        else:
            dados_numpy = dados
        dados_scaler, scaler = escalando_dados(dados_numpy, tipo=tipo)
        return dados_scaler, scaler

    def tratando_pca(
        self, dados: Union[pd.DataFrame, np.ndarray]
    ) -> tuple[PCA, np.ndarray]:
        """
        Aplica PCA aos dados.
        """
        if self.n_components is None:
            self.n_components = 6
        if isinstance(dados, pd.DataFrame):
            pd_dados_pca = dados.values
        else:
            pd_dados_pca = dados
        pca = PCA(n_components=self.n_components)
        dados_pca = pca.fit_transform(pd_dados_pca)
        return pca, dados_pca

    def tratando_RFE(
        self, dados_x: Union[pd.DataFrame, np.ndarray], dados_y: pd.DataFrame
    ) -> tuple[RFE, np.ndarray]:
        """
        Aplica RFE aos dados.

        """
        if isinstance(dados_x, pd.DataFrame):
            pd_dados_x = dados_x.values
        else:
            pd_dados_x = dados_x

        dados_x = pd_dados_x.reshape(-1, 1) if dados_x.ndim == 1 else dados_x
        dados_y = (
            dados_y.reshape(-1, 1) if dados_y.ndim == 1 else dados_y  # type:ignore
        )  # type:ignore

        rfe_model = RFE(
            estimator=LinearRegression(), n_features_to_select=self.n_features_to_select
        )
        dados_rfe = rfe_model.fit_transform(pd_dados_x, dados_y)
        return rfe_model, dados_rfe

    def tratando_variancia(
        self, dados: Union[pd.DataFrame, np.ndarray]
    ) -> tuple[VarianceThreshold, np.ndarray]:
        """
        Aplica variancia para reduzir aos dados.
        """
        if self.variancia_threshold is None:
            self.variancia_threshold = 0.1
        variancia_model = VarianceThreshold(threshold=self.variancia_threshold)
        if isinstance(dados, pd.DataFrame):
            pd_dados = dados.values
        else:
            pd_dados = dados
        dados_variancia = variancia_model.fit_transform(pd_dados)
        return variancia_model, dados_variancia

    def tratando_smart_correlation(
        self, dados: Union[pd.DataFrame, np.ndarray]
    ) -> tuple[SmartCorrelatedSelection, np.ndarray]:
        """
        Aplica smart correlation para reduzir aos dados.
        """
        if self.smart_correlated_threshold is None:
            self.smart_correlated_threshold = 0.7

        smart_correlation_model = SmartCorrelatedSelection(
            selection_method="variance", threshold=self.smart_correlated_threshold
        )
        dados_smart_correlation = smart_correlation_model.fit_transform(dados)

        return smart_correlation_model, dados_smart_correlation

    def diferenciacao_log(
        self, dados: pd.DataFrame, variavel_predicao: str
    ) -> pd.DataFrame:
        for i in dados.columns:
            if i == variavel_predicao:
                dados[i] = dados[i]
            else:
                dados[i] = np.log(dados[i] + np.abs(np.min(dados[i])) + 1)
        return dados

    def tratando_dados(
        self,
        treino_teste: bool = True,
    ) -> tuple[
        Union[np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix],
        Union[np.ndarray, pd.DataFrame, scipy.sparse.csr_matrix],
        np.ndarray,
        np.ndarray,
        Any,
        Any,
        Any,
        Any,
        Any,
    ]:

        """
        Executa todas as etapas de tratamento de dados em ordem.
        """
        # Divisão inicial de treino e teste
        treino, teste = self.tratando_divisao(self.df, treino_teste=treino_teste)

        # Aplicação das etapas de tratamento de dados
        if self.covid:

            treino = self.tratando_covid(treino)
            teste = self.tratando_covid(teste)
        if self.estacionaridade:
            if self.coluna_label is None:
                self.coluna_label = "selic"
            treino = self.tratando_estacionaridade(
                treino, coluna_label=self.coluna_label
            )
            teste = self.tratando_estacionaridade(teste, coluna_label=self.coluna_label)
        if self.estacionaridade_log:
            if self.coluna_label is None:
                self.coluna_label = "selic"
            treino = self.diferenciacao_log(treino, self.coluna_label)
            teste = self.diferenciacao_log(teste, self.coluna_label)
        if self.datas:
            treino = self.tratando_datas(treino)
            teste = self.tratando_datas(teste)
        if self.defasagens:
            treino = self.tratando_defasagens(treino)
            teste = self.tratando_defasagens(teste)

        # Separação das variáveis independentes e dependentes
        if self.coluna_label is None:
            self.coluna_label = "selic"
        x_treino, y_treino = self.tratando_divisao_x_y(treino, label=self.coluna_label)
        x_teste, y_teste = self.tratando_divisao_x_y(teste, label=self.coluna_label)

        # Escalonamento dos dados
        if self.scaler:
            x_treino, self.scaler_modelo = self.tratando_scaler(x_treino)
            x_teste = self.scaler_modelo.transform(x_teste)

        if self.variancia:
            self.variancia_modelo, x_treino = self.tratando_variancia(x_treino)
            x_teste = self.variancia_modelo.transform(x_teste)

        if self.smart_correlation:
            self.smart_correlation_modelo, x_treino = self.tratando_smart_correlation(
                x_treino
            )
            x_teste = self.smart_correlation_modelo.transform(x_teste)  # type:ignore

        if self.rfe:
            self.rfe_modelo, x_treino = self.tratando_RFE(x_treino, y_treino)
            x_teste = self.rfe_modelo.transform(x_teste)

        # Aplicação do PCA
        if self.pca:
            self.pca_modelo, x_treino = self.tratando_pca(x_treino)
            x_teste = self.pca_modelo.transform(x_teste)

        return (
            x_treino,
            x_teste,
            y_treino,
            y_teste,
            self.pca_modelo if self.pca else None,
            self.scaler_modelo if self.scaler else None,
            self.rfe_modelo if self.rfe else None,
            self.variancia_modelo if self.variancia else None,
            self.smart_correlation_modelo if self.smart_correlation else None,
        )

    def dados_futuros(
        self,
        dados_entrada: pd.DataFrame,
        ultimas_colunas: int = -10,
    ) -> pd.DataFrame:
        dados = dados_entrada.copy()
        if self.covid:
            dados = self.tratando_covid(dados_entrada)
        if self.estacionaridade:
            dados = self.tratando_estacionaridade(dados)
        if self.estacionaridade_log:
            dados = self.diferenciacao_log(dados, self.coluna_label)  # type:ignore
        if self.datas:
            dados = self.tratando_datas(dados)
        if self.defasagens:
            dados = self.tratando_defasagens(dados)
        dados = dados.drop(self.coluna_label, axis=1)
        dados = dados.iloc[ultimas_colunas:]
        if self.scaler:
            if self.scaler_modelo is not None:
                dados = self.scaler_modelo.transform(dados)
        if self.variancia:
            if self.variancia_modelo is not None:
                dados = self.variancia_modelo.transform(dados)
        if self.smart_correlation:
            if self.smart_correlation_modelo is not None:
                dados = self.smart_correlation_modelo.transform(dados)  # type:ignore
        if self.rfe:
            if self.rfe_modelo is not None:
                dados = self.rfe_modelo.transform(dados)
        if self.pca:
            if self.pca_modelo is not None:
                dados = self.pca_modelo.transform(dados)
        if not isinstance(dados, pd.DataFrame):
            if isinstance(dados, scipy.sparse.csr_matrix):
                dados = pd.DataFrame(dados.toarray(), index=dados.index)
            else:
                dados = pd.DataFrame(dados)
        return dados
