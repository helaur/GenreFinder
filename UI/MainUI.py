from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from UI.CNNUI import CNN_UI
from PyQt5 import QtCore


import numpy  as  np



class MatplotlibWidget(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)

        loadUi("kasutajaliides.ui", self)

        self.setWindowTitle("Genre Finder")
        self.pushButton.clicked.connect(self.update_text)
        self.pushButton_2.clicked.connect(self.button_pushed)
        self.cnn = CNN_UI()


        image_path = 'clef.png'
        image_profile = QImage(image_path)
        image_profile = image_profile.scaled(500, 500, aspectRatioMode=QtCore.Qt.KeepAspectRatio)


        self.label.setPixmap(QPixmap.fromImage(image_profile))

    def update_text(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file',
                                                'c:\\', "*.wav")
        self.lineEdit.setText(fname[0])



    def update_graph(self, sizes):



        labels = self.cnn.genres
        colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'blue']
        explode = [0, 0, 0, 0, 0]
        explode[np.argmax(sizes)] = 0.15
        explode = tuple(explode)
        self.widget.canvas.axes.clear()

        self.widget.canvas.axes.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)

        self.widget.canvas.draw()

    def button_pushed(self):

        print(self.lineEdit.text())
        self.update_graph(self.cnn.find_songs_genre(r"%s" % self.lineEdit.text()))

app = QApplication([])
window = MatplotlibWidget()
window.show()
app.exec_()