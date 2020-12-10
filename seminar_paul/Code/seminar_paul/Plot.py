import matplotlib.pyplot as plt

#Eine Sequenz plotten
def plot_easy(self, data):
        plt.plot(data)
        plt.show()

#Zwei sequenzen plotten
def plot_easy_two(self, data, data2):
    plt.plot(data,  label='True',      color='blue')
    plt.plot(data2, label='Predicted', color='red')
    plt.legend()
    #plt.title('Ausgangsmodell')
    plt.xlabel('Tage')
    plt.ylabel('Kurs')
    plt.show()

#Plotten der unterschiedlichen Portfolios
def plot_easy_six(self, p1, p2, p3,p4,p5,p6):
    plt.plot(p1, label='Buy&Hold',   zorder=1)
    plt.plot(p2, label='Trendfolge', zorder=-1 )
    plt.plot(p3, label='NN0',        zorder=-1 )
    plt.plot(p4, label='NN0.5',      zorder=-1)
    plt.plot(p5, label='NN1',        zorder=-1 )
    plt.plot(p6, label='NN2',        zorder=-1 )
    plt.xlabel('Tage')
    plt.ylabel('Kurs')
    plt.legend()
   #plt.title('Portfolio - Baisse')
    plt.show()

def plot_loop(self, data):
    for d in data:
        plt.plot(d)
    plt.show()

#Biepsielhafte Vorhersagen des neuronalen Netzes
def plot_net_simple(self, data, net):
    for past, future in data.take(5):
        past_size = range(-30, 0)
        future_size = range(0, 1)

        plt.plot(past_size, past[0])
        plt.plot(future_size, net.predict(past)[0], "x")
        plt.plot(future_size, future[0], "o")
        plt.show()

