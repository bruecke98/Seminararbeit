from seminar_paul import Plot, Daten, NeuralNet, Strategie
import numpy as np

class Test:

    #Einteilung der Daten in truepositive, truenegative, falsepositive, falsenegative
    def pos_neg(self, true, predicted):
        i = 0
        tn = 0
        fn = 0
        tp = 0
        fp = 0
        while (i<len(true)-1):
            if (true[i] > predicted[i+1]):
                if (true[i] > true[i+1]):
                    tn = tn +1
                if (true[i] < true[i+1]):
                    fn = fn +1
            if (true[i] < predicted[i+1]):
                if (true[i] < true[i+1]):
                    tp = tp+1
                if (true[i] > true[i+1]):
                    fp = fp +1
            i=i+1
        print("PPP", tp," ", tn," ", fp," ", fn)
        return tp, tn, fp, fn

    #Precision
    def P(self, tp, fn):
        return tp/(tp+fn)

    #Recall
    def R(self, tp, fp):
        try:
            return tp/(tp+fp)
        except: ZeroDivisionError

    #Accuracy
    def A(self, tp, tn, fp, fn):
        return (tp+tn)/(tp+tn+fp+fn)

    #F1
    def F1(self, p, r):
        try:
            return 2 * ((p*r)/(p+r))
        except TypeError:
            return 0
    #MSE
    def mse(self, true, predicted):
       # return metrics.mean_squared_error(true, predicted)
        i=0
        sum = 0
        while (i<len(true)):
            sum = sum + ((true[i]-predicted[i]) ** 2)
            i = i + 1
        mse = (1/len(true)) * sum
        return mse

    #MAE
    def mae(self, true, predicted):
        # return metrics.mean_squared_error(true, predicted)
        i=0
        sum = 0
        while (i < len(true)):
            sum = sum + abs(true[i]-predicted[i])
            i = i + 1
        mse = (1/len(true)) * sum
        return mse

    #Auswahl der Daten welche zum Testen genutzt werden
    def test_data(self, data):

        #Daten für Strategien
        hausse =  data[21200:21200+350]
        baisse =  data[18200:18200+350]
        crash =   data[20100:20100+350]
        moderat = data[22500:22500+350]


        #Testdaten
        test = data[len(data)-760:len(data)-410]

        return test



    #Skalierung der Trainingsdaten
    def scale(self, data):
        data_scal = Daten.Data.scaler_transform(Daten, data)
        factor = Test.factor(Test, data[0], data_scal[0])
        data_scal_past, data_scal_future = Daten.Data.order(Daten, data_scal, 30, 1, 0, len(data_scal))
        data_scal2 = NeuralNet.NeuralNet.processing_Vald(NeuralNet, data_scal_past, data_scal_future)
        return data_scal2, factor

    #Faktor zur Berechnung in Normale Werte
    def factor(self, orig, scal):
        return orig/scal

    #test Main Klasse
    def test(self, neuralnet, data):
        testdata = Test.test_data(Test, data)
        testdata_scale, factor = Test.scale(Test, testdata)

        for p, f in testdata_scale.take(1):
            i = 1
            #Arrays erstellen für die Wahren und Vorhergesagten Werte
            pred = np.array(neuralnet.predict(p)[i][0]*factor)
            fut = np.array(f[0]*factor)
            while (i<300):
                pred = np.append(pred, neuralnet.predict(p)[i]*factor)
                fut = np.append(fut, f[i]*factor)
                i=i+1

            #Plot der Vorhersagen im Vergleich mit den wahren Werten
            Plot.plot_easy_two(Plot, fut, pred)
            Strategie.high_low(fut)

            #truepositive, truenegative, falsepoitive, falsenegative
            tp, tn, fp, fn = Test.pos_neg(Test, fut, pred)


            print("Ergebnisse")
            print("P: ",   Test.P(Test, tp, fn))
            print("R: ",   Test.R(Test, tp, fp))
            print("A: ",   Test.A(Test, tp, tn, fp, fn))
            print("F1: ",  Test.F1(Test, Test.P(Test, tp, fn), Test.R(Test, tp, fp)))
            print("MSE: ", Test.mse(Test, fut, pred))
            print("MAE: ", Test.mae(Test, fut, pred))
            print('\n\n')


            #unterschiedliche Strategien Testen und veranschaulichen
            p1 = Strategie.Strategie.startegie_bh(Strategie, fut)
            print('\n\n')
            p2 = Strategie.Strategie.strategie_keep(Strategie, fut)
            print('\n\n')
            p3 = Strategie.Strategie.strategie(Strategie, fut, pred)
            print('\n\n')
            #SW 0.5%
            p4 = Strategie.Strategie.strategie_with_SW(Strategie, fut, pred, 0.005)
            print('\n\n')
            #SW 1%
            p5 = Strategie.Strategie.strategie_with_SW(Strategie, fut, pred, 0.01)
            print('\n\n')
            #SW 2%
            p6 = Strategie.Strategie.strategie_with_SW(Strategie, fut, pred, 0.02)
            print('\n\n')

            #Plot aller Strategien
            Plot.plot_easy_six(Plot, p1, p2, p3, p4, p5, p6)






    #AKtuelle Vorhersagen
    #Aktuelle Daten
    def aktuelle_daten(self, data):
        aktuell = data[len(data)-31:len(data)]
        return aktuell

    #Aktuelle Daten skalieren und Preprocessed
    def scale_aktuelledaten(self, data):
        data_scal = Daten.Data.scaler_transform(Daten, data)
        factor = Test.factor(Test, data[0], data_scal[0])
        data_scal_past, data_scal_future = Daten.Data.order(Daten, data_scal, 30, 0, 0, len(data_scal)+2)
        data_scal2 = NeuralNet.NeuralNet.processing_Vald(NeuralNet, data_scal_past, data_scal_future)
        return data_scal2, factor

    def prediction(self, neuralnet, data):
        prediction = 0
        last_price = 0
        testdata = Test.aktuelle_daten(Test, data)
        testdata_scale, factor = Test.scale_aktuelledaten(Test, testdata)
        print('Vorhersage')
        print(testdata_scale)
        for p, f in testdata_scale.take(1):
            #print(p)
            last_price = p[0][29]*factor
            prediction = neuralnet.predict(p)[0]*factor
            i=0
            print("Stimmt?")
            print(p[i]*factor)
            #print(f[i]*factor)
            print(last_price)
            print(prediction)

            #print(f)

        print('prognostizierte Kursentwicklung Morgen: ', ((prediction/last_price)-1))
