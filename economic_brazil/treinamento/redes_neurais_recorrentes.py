import keras
import matplotlib.pyplot as plt
import keras_tuner
import pandas as pd
import numpy as np


class RnnModel:
    def create_dataset(self, data_x, data_y, time_step=1):
        X, Y = [], []
        for i in range(len(data_x) - time_step):
            a = data_x[i : (i + time_step)]
            X.append(a)
            Y.append(data_y[i + time_step])
        return np.array(X), np.array(Y)

    def create_rnn_model(
        self,
        input_shape,
        num_layers=2,
        units=50,
        dropout_rate=0.2,
        num_outputs=1,
        optimizer="adam",
        loss="mean_squared_error",
        activation="relu",
        metrics=None,
    ):
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
            metrics = ["accuracy"]
        model = keras.Sequential()
        # Adicionando a primeira camada LSTM
        model.add(
            keras.layers.LSTM(
                units,
                return_sequences=True if num_layers > 1 else False,
                input_shape=input_shape,
            )
        )
        model.add(keras.layers.Dropout(dropout_rate))

        # Adicionando camadas LSTM adicionais se necessário
        for i in range(1, num_layers):
            return_sequences = (
                i != num_layers - 1
            )  # A última camada LSTM não deve retornar sequências
            model.add(keras.layers.LSTM(units, return_sequences=return_sequences))
            model.add(keras.layers.Dropout(dropout_rate))

        # Camada densa para aprendizado adicional e camada de saída
        model.add(keras.layers.Dense(units, activation=activation))
        model.add(keras.layers.Dense(num_outputs))

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model
        

    def treinar_modelo(
        self,
        model,
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        test_data=None,
        X_test=None,
        y_test=None,
    ):
        if test_data:
            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
            )
        else:
            model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        return model

    def predizer_modelo(self, model, dados):
        return model.predict(dados)

    def avaliar_modelo(self, model, X_test, y_test):
        return model.evaluate(X_test, y_test)

    def plotar_grafico(self, dataset, tamanho=(10, 5)):
        history = pd.DataFrame(dataset.history.history)

        plt.figure(figsize=tamanho)
        plt.subplot(1, 2, 1)
        plt.plot(history.loss, label="Training Loss")
        plt.axvline(history.val_loss.idxmax(), ls="--", lw=1, c="k")
        plt.plot(history.val_loss, label="Validation Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if "accuracy" in history.columns:
            plt.subplot(1, 2, 2)
            plt.plot(history.accuracy, label="Training Accuracy")
            plt.plot(history.val_accuracy, label="Validation Accuracy")
            # plt.axline(history.val_accuracy.idxmax(), ls='--', lw=1, c='k')
            plt.title("Training and Validation Accuracy")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()

        plt.tight_layout()
        plt.show()

class HyperTurnerModel:
    def __init__(self, x_train, y_train, x_test, y_test,camadas_sequenciais=3):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.camadas_sequenciais = camadas_sequenciais

    def build_model(self, hp, learning_rate=None, neuronios_camada_1=None, neuronios_camada_sequenciais=None, neuronios_camada_final=1):
        if learning_rate is None:
            learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4, 1e-5, 1e-6])
        if neuronios_camada_1 is None:
            neuronios_camada_1 = hp.Int('neuronios_camada_1', min_value=8, max_value=64, step=8)
        if neuronios_camada_sequenciais is None:
            neuronios_camada_sequenciais = hp.Int('neuronios_camada_sequenciais', min_value=8, max_value=64, step=8)

        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=neuronios_camada_1,
            activation='relu',
            input_shape=(self.x_train.shape[1], self.x_train.shape[2]),
            return_sequences=True))

        for _ in range(self.camadas_sequenciais - 1):
            model.add(keras.layers.Dropout(rate=hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)))
            model.add(keras.layers.LSTM(
                units=neuronios_camada_sequenciais,
                activation='relu',
                return_sequences=True))

        model.add(keras.layers.Dense(neuronios_camada_final))
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss='mse')
        return model

    def tuner(self, epochs=100, batch_size=32):
        tuner = keras_tuner.RandomSearch(
            hypermodel=self.build_model,
            objective='val_loss',
            max_trials=5,
            executions_per_trial=3,
            overwrite=True)

        tuner.search(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_data=(self.x_test, self.y_test))
        best_model = tuner.get_best_models(num_models=1)[0]
        if best_model:
            print(best_model.summary())
        return best_model


