from seminar_paul import Daten, Plot, NeuralNet, Strategie, Test
import numpy as np
import tensorflow as tf


print(tf.version.VERSION)

#Variablen initialsisieren
future_size = 1
past_size = 30
data = Daten.Data
plot = Plot

#Laden der Daten
data_fin = Daten.Data.import_data_from_csv(data, "C:\\seminar_paul\\Code\Daten\\^GSPC.csv")

#Auswahl der Schlusskurse
dn = np.array(data_fin['Close']).reshape(-1,1)

#Daten Skalieren
d = Daten.Data.scaler_transform(data, dn)

#Data Split und aufteilen der Daten in Input und Output
datas_train_1, labels_train_1 = data.order(data, d, past_size, future_size, 0,           len(d)*0.2)
datas_vald_1,  labels_vald_1 = data.order( data, d, past_size, future_size, len(d)*0.2,  len(d)*0.25)
datas_train_2, labels_train_2 = data.order(data, d, past_size, future_size, len(d)*0.25, len(d)*0.45)
datas_vald_2,  labels_vald_2 = data.order( data, d, past_size, future_size, len(d)*0.45, len(d)*0.5)
datas_train_3, labels_train_3 = data.order(data, d, past_size, future_size, len(d)*0.5,  len(d)*0.7)
datas_vald_3,  labels_vald_3 = data.order( data, d, past_size, future_size, len(d)*0.7,  len(d)*0.75)
datas_train_4, labels_train_4 = data.order(data, d, past_size, future_size, len(d)*0.75, len(d)*0.95)
datas_vald_4,  labels_vald_4 = data.order( data, d, past_size, future_size, len(d)*0.95, len(d))

datas_train =  np.concatenate((datas_train_1,  datas_train_2,  datas_train_3,   datas_train_4))
labels_train = np.concatenate((labels_train_1, labels_train_2, labels_train_3,  labels_train_4))
datas_vald =   np.concatenate((datas_vald_1,   datas_vald_2,   datas_vald_3,    datas_vald_4))
labels_vald =  np.concatenate((labels_vald_1,  labels_vald_2,  labels_vald_3,   labels_vald_4))


#initialisiern der NeuronalNet Klasse
neuralNet_train = NeuralNet.NeuralNet(1000, 30, np.array(datas_train))
neuralNet_vald = NeuralNet.NeuralNet(1000,  30, np.array(datas_vald))


#Preprocessing der Daten
train_data = NeuralNet.NeuralNet.processing_Train(neuralNet_train,  datas_train,  labels_train)
vald_data = NeuralNet.NeuralNet.processing_Vald(neuralNet_vald,     datas_vald,   labels_vald)


#Neuroneles Netz erstellen und trainieren
#neuralNet = NeuralNet.NeuralNet.NeuralNet_model(neuralNet_train, past_size, future_size, train_data, vald_data)

#Laden vorhandener Netze
neuralNet = tf.keras.models.load_model("C:\\seminar_paul\\Models\\vierschichtig_tanh\\")


#Test und Strategie
Test.Test.test(Test, neuralNet, dn)

#Vorhersagen für den nächstn Tag erstellen
Test.Test.prediction(Test, neuralNet, dn)

#Gute Netze Speichern
speichern = input("Speichern? - Name?")
neuralNet.save("C:\\seminar_paul\\Models\\"+speichern+"\\")



#Beispielhafte Plots von Vorhersagen
plot.plot_net_simple(plot, vald_data, neuralNet)
