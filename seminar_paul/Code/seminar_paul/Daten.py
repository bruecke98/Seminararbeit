import pandas as pd
import numpy as np
from sklearn import preprocessing
from seminar_paul import Plot


class Data:
    def __init__(self):
        data = Data
        plot = Plot

    def import_data_from_csv(self, data_path):
        #Daten von einer csv Datei zu pandas Array!
        try:
            #importieren wenn es die Datei gibt
            return pd.read_csv(data_path, index_col=0, parse_dates=True).dropna()
        except FileNotFoundError:
            print('Datei nicht gefunden!')


    #def scaler_factor(self, data , scaled_data):
    #    self.factor = data[0]/scaled_data[0]
    #    return data[0]/scaled_data[0]

    def scaler_transform(self, data):
        #Einstellen welchen Scaler man nutzen will
        scaler = preprocessing.MaxAbsScaler()
        return scaler.fit_transform(data)



    def order(self,data_fin, past_size, future_size, start, end):
        data = []
        labels = []
        i = int (start) + past_size + 1

        while i<(end - future_size - 1):
            labels.append(data_fin[i:i+future_size])
            data.append(data_fin[i-past_size:i])
            i = i + 1

        return np.array(data) , np.array(labels)

