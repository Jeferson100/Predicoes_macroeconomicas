import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def treino_test_dados(dados,data_divisao,coluna):
    treino = dados[dados.index < pd.to_datetime(data_divisao)]
    y_treino = treino[coluna].values
    x_treino = treino.loc[:,treino.columns != coluna].values  
    teste = dados[dados.index >= pd.to_datetime(data_divisao)]
    y_teste = teste[coluna].values
    x_teste= teste.loc[:,teste.columns != coluna].values
    print('O tamanho do y_treino e ',y_treino.shape)
    print('O tamanho do x_treino e ',x_treino.shape)
    print('--------------------')
    print('O tamanho do y_teste e ',y_teste.shape)
    print('O tamanho do x_teste e ',x_teste.shape)
    print('--------------------')
    return x_treino,y_treino,x_teste,y_teste

#write a fuction for split data for series temporary test and train


def treino_teste_seies_temporal(dados,coluna, numero_divisoes=5,gap_series=5,max_train_size=100,test_size=10):
    y = dados[coluna]
    X = dados.drop(coluna, axis=1)
    ts_cv = TimeSeriesSplit(
    n_splits=numero_divisoes,  # to keep the notebook fast enough on common laptops
    gap=gap_series,  # 2 days data gap between train and test
    max_train_size=max_train_size,  # keep train sets of comparable sizes
    test_size=test_size)  # for 2 or 3 digits of precision in scores)
    all_splits = list(ts_cv.split(X, y))    
    return all_splits

