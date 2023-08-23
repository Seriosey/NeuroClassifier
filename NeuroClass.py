# coding=windows-1251
import random
import sys 
import os
#import PyQt6
from PyQt6 import QtWidgets, QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QFileDialog, QTableView , QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QHBoxLayout
#from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt, QAbstractTableModel
from PyQt6 import uic
from pathlib import Path
import locale
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
from PyQt6.QtGui import QIcon
import pandas as pd
from Classif import *
from preprocessing_1 import *
from preprocessing_2 import *


#locale.setlocale(locale.LC_ALL, 'ru_RU', 'UTF-8')

path = None

class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        uic.loadUi('UI0.ui', self)
        self.setWindowTitle('NeuroClassifier')
        self.setWindowIcon(QIcon('Hippo.jpg'))
        #self.setWindowIcon(QIcon('maps.ico'))
        #self.resize(800, 600)
        self.Button.clicked.connect(self.ChooseFile)
        #layout = QVBoxLayout()
        #self.setLayout(layout)
        self.Button2.clicked.connect(self.Run)
        self.Button3.clicked.connect(self.get_table)
        self.Button4.clicked.connect(self.scatter)
        self.centralwidget = QtWidgets.QWidget()
        self.file_list = QtWidgets.QListWidget(self.centralwidget)
        self.file_list.setGeometry(QtCore.QRect(0, 20, 131, 151))
        self.file_list.setObjectName("listWidget")
        self.progressBar.setValue(0)
        self.progressBar_2.setValue(0)
        self.progressBar_3.setValue(0)
        self.progressBar_4.setValue(0)
        self.progressBar_5.setValue(0)
        self.progressBar_6.setValue(0)
        self.progressBar_7.setValue(0)
        #self.path

        #widgets
        #self.inputField = QLineEdit()
        #button = QPushButton('&What type of neuron is it?')
        #button = QPushButton('&What type of neuron is it?', clicked=self.Classify)
        #
        #self.output = QTextEdit()

        '''layout.addWidget(self.inputField)
        layout.addWidget(button)
        layout.addWidget(self.output'''

    
    
    def ChooseFile(self):
        dialog = QFileDialog(self)
        dialog.setDirectory(r'C:')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dialog.setNameFilter("HDF5 (*.hdf5 )")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            filenames = dialog.selectedFiles()
            if filenames:
                #self.file_list.addItems([str(Path(filename)) for filename in filenames])
                file_path, _ = QFileDialog.getOpenFileName(None, 'Choose file', '', 'file HDF (*.hdf5)')
                global table
                table = pd.read_hdf(file_path)
                global path
                path = file_path
                return table
                      
                

    def get_table(self):
        dialog = QFileDialog(self)
        #dialog.setDirectory(r'C:')
        dialog.setFileMode(QFileDialog.FileMode.ExistingFiles)
        #dialog.setNameFilter("HDF5 (*.hdf5 )")
        dialog.setViewMode(QFileDialog.ViewMode.List)
        if dialog.exec():
            dir_name = QFileDialog.getExistingDirectory(self, "Select a Directory with source hdf5 files")
            if dir_name:
                source_path = Path(dir_name)
            dir_name2 = QFileDialog.getExistingDirectory(self, "Select a Directory for meta files")
            if dir_name2:
                target_path = Path(dir_name2)


            filenames = dialog.selectedFiles()
            if filenames:
                pass
                #self.file_list.addItems([str(Path(filename)) for filename in filenames])
                #source_path, _ = QFileDialog.getExistingDirectory(self,"Choose source directory",".")
                #source_path, _ = QFileDialog.getOpenFileName(None, 'Choose source file', '', 'file HDF (*.hdf5)')
                #target_path, _ = QFileDialog.getOpenFileName(None, 'Choose target file', '', 'file HDF (*.hdf5)')
                
        

        self.textEdit_2.setText("Параметры активности рассчитываются...")
        proc1(source_path, target_path)
        proc2(target_path)
        

        #df = pd.DataFrame(np.array(h5py.File('./feasures_table0.hdf5')['df']))
        
        table = pd.read_hdf('./feasures_table0.hdf5')
        
        
        model = pandasModel(table)
        self.tableView.setModel(model)
        self.textEdit_2.setText("Таблица с параметрами активности - <feasure_table.hdf5> - сохранена в каталоге приложения")
        

    #def Classify(self):
    #    intputText = self.Input.text()
    #    self.Output.setText('I suppose it is a {0}'.format(intputText))

    def scatter(self):
        scatterPCA(path)


    def Run(self): 
        res = list(do(path))
        #res = [10, 20, 25, 15, 29, 1]

        self.progressBar.setValue(int(res[0]*100))
        self.progressBar_2.setValue(int(res[1]*100))
        self.progressBar_3.setValue(int(res[2]*100))
        self.progressBar_4.setValue(int(res[3]*100))
        self.progressBar_5.setValue(int(res[4]*100))
        self.progressBar_6.setValue(int(res[5]*100))
        self.progressBar_7.setValue(int(res[6]*100))
        types= ['NGL', 'OLM', 'AAC', 'IVY', 'BSC', 'BC', 'CCKBAS']
        

        self.textEdit.setText('In given dataset most of neurons - {0}% are {1} neurons'.format(max(res)*100, types[np.argmax(res)]))


        self.graphWidget = pg.PlotWidget(self.tab_5)

        #scatterPCA()

      


class pandasModel(QAbstractTableModel):

    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parnet=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, col, orientation, role):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._data.columns[col]
        return None
        
if __name__ == '__main__':
    app = QApplication(sys.argv)

    '''
    app.setStyleSheet(
    QWidget{
        font-size:25px;
    }
    QPushButton{
    font-size: 20px;}

    QPlainTextEdit{
    font-size: 15px}
    )'''
    

    file = open("./Genetive/Genetive.qss",'r')

    with file:
        qss = file.read()
        app.setStyleSheet(qss)


    myApp = MyApp()
    myApp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window')


