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


class TratandoDados:
    def __init__(self, df):
        self.df = df
        self.scaler_modelo = None
        self.pca_modelo = None

    # pylint: disable=W0632
    def tratando_divisao(
        self, dados, divisao_treino_teste="2020-04-01", treino_teste=True
    ):
        """
        Divide os dados em conjuntos de treino e teste.
        """
        if treino_teste:
            # pylint: disable=W0632
            treino, teste = treino_test_dados(
                dados, data_divisao=divisao_treino_teste, treino_teste=True
            )
            
            return treino, teste
        else:
            # pylint: disable=W0632
            x_treino, y_treino, x_teste, y_teste = treino_test_dados(
                dados, data_divisao=divisao_treino_teste
            )  
            return x_treino, y_treino, x_teste, y_teste
    # pylint: disable=W0632
    def tratando_covid(
        self, dados, inicio_periodo="2020-04-01", fim_periodo="2020-05-01"
    ):
        """
        Adiciona variáveis dummy para o período COVID.
        """
        dados_covid = criando_dummy_covid(
            dados, inicio_periodo=inicio_periodo, fim_periodo=fim_periodo
        )
        return dados_covid

    def tratando_estacionaridade(self, dados, coluna_label="selic"):
        """
        Corrige a não-estacionaridade dos dados.
        """
        estacionaridade = Estacionaridade()
        dados_est = estacionaridade.corrigindo_nao_estacionaridade(dados, coluna_label)
        return dados_est

    def tratando_datas(self, dados, mes=True, trimestre=True, dummy=True, colunas=None):
        """
        Adiciona colunas de mês, trimestre e dummies aos dados.
        """
        if colunas is None:
            colunas = ["mes", "trimestre"]
        dados_datas = criando_mes_ano_dia(
            dados, mes=mes, trimestre=trimestre, dummy=dummy, coluns=colunas
        )
        return dados_datas

    def tratando_defasagens(self, dados, numero_defasagens=4):
        """
        Cria defasagens nos dados.
        """
        dados_defas = criando_defasagens(dados, numero_defasagens=numero_defasagens)
        return dados_defas[numero_defasagens:]

    def tratando_divisao_x_y(self, dados, label="selic"):
        """
        Separa as variáveis independentes e dependentes.
        """
        y = dados[label].values
        x = dados.loc[:, dados.columns != label].values
        return x, y

    def tratando_scaler(self, dados, tipo="scaler"):
        """
        Escala os dados usando o método especificado.
        """
        dados_scaler, scaler = escalando_dados(dados, tipo=tipo)
        return dados_scaler, scaler

    def tratando_pca(self, dados, n_components=6):
        """
        Aplica PCA aos dados.
        """
        pca = PCA(n_components=n_components)
        dados_pca = pca.fit_transform(dados)
        return pca, dados_pca

    def tratando_dados(
        self,
        treino_teste=True,
        covid=True,
        estacionaridade=True,
        datas=True,
        defasagens=True,
        pca=True,
        scaler=True,
    ):
        """
        Executa todas as etapas de tratamento de dados em ordem.
        """
        # Divisão inicial de treino e teste
        treino, teste = self.tratando_divisao(self.df, treino_teste=treino_teste)

        # Aplicação das etapas de tratamento de dados
        if covid:
            treino = self.tratando_covid(treino)
            teste = self.tratando_covid(teste)
        if estacionaridade:
            treino = self.tratando_estacionaridade(treino)
            teste = self.tratando_estacionaridade(teste)
        if datas:
            treino = self.tratando_datas(treino)
            teste = self.tratando_datas(teste)
        if defasagens:
            treino = self.tratando_defasagens(treino)
            teste = self.tratando_defasagens(teste)

        # Separação das variáveis independentes e dependentes
        x_treino, y_treino = self.tratando_divisao_x_y(treino)
        x_teste, y_teste = self.tratando_divisao_x_y(teste)

        # Escalonamento dos dados
        if scaler:
            x_treino, self.scaler_modelo = self.tratando_scaler(x_treino)
            x_teste = self.scaler_modelo.transform(x_teste)

        # Aplicação do PCA
        if pca:
            self.pca_modelo, x_treino = self.tratando_pca(x_treino)
            x_teste = self.pca_modelo.transform(x_teste)

        return x_treino, x_teste, y_treino, y_teste, self.pca_modelo, self.scaler_modelo

    def dados_futuros(
        self,
        dados_entrada,
        covid=True,
        estacionaridade=True,
        datas=True,
        defasagens=True,
        pca=True,
        scaler=True,
        coluna_drop="selic",
        ultimas_colunas=-10,
    ):
        if covid:
            dados = self.tratando_covid(dados_entrada)
        if estacionaridade:
            dados = self.tratando_estacionaridade(dados)
        if datas:
            dados = self.tratando_datas(dados)
        if defasagens:
            dados = self.tratando_defasagens(dados)
        dados = dados.drop(coluna_drop, axis=1)
        dados = dados.iloc[ultimas_colunas:]
        if scaler:
            dados = self.scaler_modelo.transform(dados)
        if pca:
            dados = self.pca_modelo.transform(dados)
        return dados
