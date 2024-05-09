import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from economic_brazil.processando_dados.divisao_treino_teste import treino_test_dados
from economic_brazil.processando_dados.estacionaridade import Estacionaridade
from economic_brazil.processando_dados.data_processing import criando_dummy_covid, escalando_dados, criando_mes_ano_dia
import shap

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


class TratandoDados:
    def __init__(self, df):
        self.df = df 
        
    def tratando_dados(self, colunas_label='selic',divisao_treino_teste='2020-04-01'):
        treino, teste = treino_test_dados(self.df,data_divisao=divisao_treino_teste,treino_teste=True)
        estacionaridade = Estacionaridade()
        ##Dados treino
        train_covid = criando_dummy_covid(treino,inicio_periodo='2020-04-01', fim_periodo='2020-05-01')
        train_est = estacionaridade.corrigindo_nao_estacionaridade(train_covid,colunas_label)
        train_datas = criando_mes_ano_dia(train_est,mes=True,trimestre=True,dummy=True,coluns=['mes','trimestre'])
        y_train = train_datas[colunas_label].values
        columns = train_datas.loc[:, train_datas.columns != colunas_label].columns
        x_train = train_datas.loc[:, train_datas.columns != colunas_label].values
        x_train_scaler, scaler = escalando_dados(x_train,tipo="scaler") 
        #Dados teste
        test_covid = criando_dummy_covid(teste,inicio_periodo='2020-04-01', fim_periodo='2020-05-01')
        test_est = estacionaridade.corrigindo_nao_estacionaridade(test_covid,colunas_label)
        test_datas = criando_mes_ano_dia(test_est,mes=True,trimestre=True,dummy=True,coluns=['mes','trimestre'])
        y_test= test_datas[colunas_label].values
        x_test = test_datas.loc[:, test_datas.columns != colunas_label].values
        x_test_scaler = scaler.transform(x_test)
        
        return x_train_scaler, y_train, x_test_scaler, y_test, columns

class ImportanciaRandomForest:
    def __init__(self, X, y,colunas, model=None):
        self.X = X
        self.y = y
        self.model = model
        self.colunas = colunas
    
    def treinar_modelo(self):
        """Treina o modelo RandomForestRegressor com os dados fornecidos."""
        model = RandomForestRegressor()
        self.model = model.fit(self.X, self.y)
        return self.model

    
    def importancia_caracteristicas(self, plot=True, tamanho=(14, 6)):
        """Retorna a importância das características do modelo RandomForest treinado."""
        if self.model is None:
            self.treinar_modelo()
        
        feature_importances = pd.DataFrame(
            self.model.feature_importances_,
            index=self.colunas,
            columns=['importance']
        ).sort_values('importance', ascending=False)
        
        if plot:
            plt.figure(figsize=tamanho)
            sns.barplot(x="importance", y=feature_importances.index, data=feature_importances)
            plt.xlabel('Importância')
            plt.ylabel('Característica')
            plt.title('Importância das Características - RandomForest',fontsize=14)
            plt.tight_layout()
            plt.subplots_adjust(top=.9);
            plt.show()
            
        else:
            return feature_importances


        
class ImportanciaShap:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.model = None
    
    def treinar_modelo(self):
        """Treina o modelo RandomForestRegressor com os dados fornecidos."""
        if self.model is None:
            self.model = RandomForestRegressor()
            self.model.fit(self.X, self.y)
        else:
            raise RuntimeError("Modelo já foi treinado.")
        
    def importancia_shap(self,columns, plot=True):
        model = self.model
        explainer = shap.TreeExplainer(model)
        


