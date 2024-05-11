from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class RnnModel:
    def create_dataset(self,data_x,data_y, time_step=1):
        X, Y = [], []
        for i in range(len(data_x) - time_step):
            a = data_x[i:(i + time_step)]  
            X.append(a)
            Y.append(data_y[i + time_step])  
        return np.array(X), np.array(Y)
        

    def create_rnn_model(self,input_shape, num_layers=2, units=50, dropout_rate=0.2, num_outputs=1, optimizer='adam', loss='mean_squared_error',activation='relu',metrics=None):
        """
        Cria uma rede neural recorrente com um número configurável de camadas LSTM.

        :param input_shape: Tupla que define a forma da entrada, (time_steps, num_features)
        :param num_layers: Número de camadas LSTM na rede
        :param units: Número de unidades nas camadas LSTM
        :param dropout_rate: Taxa de dropout para reduzir o overfitting
        :param num_outputs: Número de unidades na camada de saída
        :return: Objeto do modelo Keras compilado
        """
        if metrics is None:
            metrics = ['accuracy']
        model = Sequential()
        # Adicionando a primeira camada LSTM
        model.add(LSTM(units, return_sequences=True if num_layers > 1 else False, input_shape=input_shape))
        model.add(Dropout(dropout_rate))

        # Adicionando camadas LSTM adicionais se necessário
        for i in range(1, num_layers):
            return_sequences = i != num_layers - 1  # A última camada LSTM não deve retornar sequências
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(dropout_rate))

        # Camada densa para aprendizado adicional e camada de saída
        model.add(Dense(units, activation=activation))
        model.add(Dense(num_outputs))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
    
    def treinar_modelo(self, model, X_train, y_train, epochs=100, batch_size=32,test_data=None, X_test=None, y_test=None):
        if test_data:
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
        else:
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return model
    
    def predizer_modelo(self, model, dados):
        return model.predict(dados)
    
    def avaliar_modelo(self, model, X_test, y_test):
        return model.evaluate(X_test, y_test)
    
    def plotar_grafico(self,dataset,tamanho=(10,5)):
        history = pd.DataFrame(dataset.history.history)
            

        plt.figure(figsize=tamanho)
        plt.subplot(1, 2, 1)
        plt.plot(history.loss, label='Training Loss')
        plt.axvline(history.val_loss.idxmax(), ls='--', lw=1, c='k')
        plt.plot(history.val_loss, label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        if 'accuracy' in history.columns:
            plt.subplot(1, 2, 2)
            plt.plot(history.accuracy, label='Training Accuracy')
            plt.plot(history.val_accuracy, label='Validation Accuracy')
            #plt.axline(history.val_accuracy.idxmax(), ls='--', lw=1, c='k')
            plt.title('Training and Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()

        plt.tight_layout()
        plt.show()
