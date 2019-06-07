import pickle
import matplotlib.pyplot as plt


class ReadModelHistory():
    def __init__(self, path):
        history_file = open(path, 'rb')
        self.history = pickle.load(history_file)



    def read_loss(self):
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Mudeli kadu')
        plt.ylabel('Kadu')
        plt.xlabel('Epohh')
        plt.legend(['Treenimine', 'Valideerimine'], loc='upper left')
        plt.savefig('loss_10s_model_200_epoch.png')
        plt.show()

    def read_acc(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Mudeli täpsus')
        plt.ylabel('Täpsus')
        plt.xlabel('Epohh')
        plt.legend(['Treenimine', 'Valideerimine'], loc='upper left')
        plt.savefig('accuracy_10s_model_200_epoch.png')
        plt.show()


history = ReadModelHistory(r"..\saved_models\history\10\model-acc 0.7653- loss 0.7531 - history")

history.read_loss()
history.read_acc()