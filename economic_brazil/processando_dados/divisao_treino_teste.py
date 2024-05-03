import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit


def treino_test_dados(dados, data_divisao, coluna=None, treino_teste=None, tipo=None):
    treino = dados[dados.index < pd.to_datetime(data_divisao)]
    teste = dados[dados.index >= pd.to_datetime(data_divisao)]
    if treino_teste:
        print("O tamanho do treino e ", treino.shape)
        print("O tamanho do teste e ", teste.shape)
        return treino, teste
    else:
        if tipo:
            y_treino = treino[coluna].values
            x_treino = treino.loc[:, treino.columns != coluna].values
            y_teste = teste[coluna].values
            x_teste = teste.loc[:, teste.columns != coluna].values
        else:
            y_treino = treino[coluna]
            x_treino = treino.loc[:, treino.columns != coluna]
            y_teste = teste[coluna]
            x_teste = teste.loc[:, teste.columns != coluna]

        print("O tamanho do y_treino e ", y_treino.shape)
        print("O tamanho do x_treino e ", x_treino.shape)
        print("--------------------")
        print("O tamanho do y_teste e ", y_teste.shape)
        print("O tamanho do x_teste e ", x_teste.shape)
        print("--------------------")
        return x_treino, y_treino, x_teste, y_teste


# write a fuction for split data for series temporary test and train


def treino_teste_seies_temporal(
    x_train,
    y_train,
    coluna,
    numero_divisoes=5,
    gap_series=5,
    max_train_size=100,
    test_size=10,
):
    ts_cv = TimeSeriesSplit(
        n_splits=numero_divisoes,  # to keep the notebook fast enough on common laptops
        gap=gap_series,  # 2 days data gap between train and test
        max_train_size=max_train_size,  # keep train sets of comparable sizes
        test_size=test_size,
    )  # for 2 or 3 digits of precision in scores)
    all_splits = list(ts_cv.split(x_train, y_train))
    return all_splits
