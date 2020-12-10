import tensorflow as tf
import matplotlib.pyplot as plt


class NeuralNet(object):
    def __init__(self, buffer, batch, data):
        self.buffer = buffer
        self.batch = batch
        self.data = data

    def processing_single(self,  data):
        #input Pipeline, apply Dataset to preprcess the data
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.batch(400).repeat()
        return data

    def processing_Train(self,  past_data, future_data):
        #input Pipeline, apply Dataset to preprcess the data
        data = tf.data.Dataset.from_tensor_slices((past_data, future_data))
        data = data.shuffle(10000).batch(400).cache().repeat()
        return data

    def processing_Vald(self,  past_data, future_data):
        #input Pipeline, apply Dataset to preprcess the data
        data = tf.data.Dataset.from_tensor_slices((past_data, future_data))
        data = data.batch(400).repeat()
        return data

    #Visualisierung des Trainings
    def visualize_net(self, fitting):
        fit = fitting.history
        plt.plot(fit['val_loss'])
        plt.plot(fit['loss'])
        plt.show()

    def NeuralNet_model(self, past_size, future_size, train_data, vald_data):
        #Initialisierung des Neuronalen Netzes
        model = tf.keras.models.Sequential()
        #Eingabeschicht
        model.add(tf.keras.layers.LSTM(past_size, return_sequences='true', activation='tanh'))
        #model.add(tf.keras.layers.LSTM(past_size, activation='tanh'))

        #Hidden Layers
        model.add(tf.keras.layers.LSTM(15, return_sequences='true', activation='tanh'))
        model.add(tf.keras.layers.LSTM(15))

        #Ausgabeschicht
        model.add(tf.keras.layers.Dense(future_size))
        model.compile(optimizer='adam', loss='MAE')

        #Training
        fitting = model.fit(train_data,
                            epochs=20,
                            steps_per_epoch=200,
                            validation_data=vald_data,
                            validation_steps=100)
        model.summary()
        NeuralNet.visualize_net(NeuralNet, fitting)
        return model







