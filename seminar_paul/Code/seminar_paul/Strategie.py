import numpy as np
from numpy import *



class Strategie():
    def beste(self, true):
        i=1
        start = 10000
        port = np.array(10000)
        while (i<(len(true)-1)):
            if (true[i+1]>true[i]):
                port = np.append(port, start*(true[i+1]/true[i]))
                start = start * (true[i+1]/true[i])
            else:
                port = np.append(port, start)
            i=i+1
        print('Rendite nur positiv: ', ((start/10000) -1))
        Strategie.vola(Strategie, port)
        Strategie.drawdown(Strategie, port)
        #Plot.plot_easy(Plot, port)


    def startegie_bh(self, true):
        i=1
        port = np.array(10000)
        start = 10000
        while (i<len(true)-1):
            port =  np.append(port, start * (true[i+1]/true[i]))
            start = start * (true[i+1]/true[i])
            i=i+1
        #Plot.plot_easy(Plot, port)
        print('Rendite Buy Hold', (true[len(true)-1]/true[0]) -1 )
        Strategie.vola(Strategie, port)
        Strategie.drawdown(Strategie, port)
        return port

    def strategie(self, true, pred):
        start = 10000
        port = np.array(10000)
        i = 1
        while (i<len(true)-1):
            if (pred[i+1]>true[i]):
                port = np.append(port, start * (true[i+1]/true[i]))
                start = start * (true[i+1]/true[i])
            else:
                port = np.append(port, start)
            i=i+1
        print('Rendite ohne Schwellwert: ', ((start/10000) -1))
        Strategie.vola(Strategie, port)
        Strategie.drawdown(Strategie, port)
        #Plot.plot_easy(Plot, port)
        return port

    def strategie_with_SW(self, true, pred, perc):
        start = 10000
        port = np.array(10000)
        i = 1
        while (i<len(true)-1):
            if ((pred[i+1]*(1+perc))>true[i]):
                port = np.append(port, start * (true[i+1]/true[i]))
                start = start * (true[i+1]/true[i])
            else:
                port = np.append(port, start)
            i=i+1
        print('Rendite mit ', perc, ' Schwellwert: ', ((start/10000) -1) )
        Strategie.vola(Strategie, port)
        Strategie.drawdown(Strategie, port)
        #Plot.plot_easy(Plot, port)
        return port


    def strategie_keep(self, true):
        start = 10000
        port = np.array(10000)
        i = 0
        while (i<len(true)-2):
            if (true[i]<true[i+1]):
                port = np.append(port, start * (true[i+2]/true[i+1]))
                start = start * (true[i+2]/true[i+1])
            else:
                port = np.append(port, start)
            i=i+1
        print('Rendite Trendfolge: ', ((start/10000) -1))
        Strategie.vola(Strategie, port)
        Strategie.drawdown(Strategie, port)
        return port
        #Plot.plot_easy(Plot, port)

    def drawdown(self, portfolio):
        i=1
        drawdown = 1
        hoch = 10000
        tief = 10000
        hoch_tag = 0
        tief_tag = 0
        max = np.array(portfolio[0])
        while (i<len(portfolio)):
            if (portfolio[i]>hoch):
                hoch = portfolio[i]
                max = np.append(max, hoch)
            else:
                max = np.append(max, hoch)
            i=i+1
        i=0
        while (i<len(portfolio)):
            if (drawdown>portfolio[i]/max[i]):
                drawdown = portfolio[i]/max[i]
                tief_tag = i
                tief = portfolio[i]
                hoch = max[i]
            i=i+1


        print('max Drawdown: ', drawdown -1 )
        #Plot.plot_easy_two(Plot, portfolio, max)


    #Volatilität/Standardabweichung
    def vola(self, portfolio):
        print('Vola ', portfolio.std())


#Methode wie viele negative und positive Börsentage in einem Datensatz enthalten sind
def high_low(data):
    high =0
    low =0
    ges= 0
    last = 0
    for d in data:
        if (d>last):
            high=high+1
        else:
            low=low+1
        last=d
        ges=ges+1
    print("Gesamt: ", ges)
    print("Positive Entwicklung ", high/ges)
    print("Negative Entwicklung ", low/ges)