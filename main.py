"""
Main module 
"""

import sys
from PyQt4 import QtGui, QtCore
from gui import ImageWin

def main():
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_X11InitThreads)
    app = QtGui.QApplication(sys.argv)
    window = ImageWin()
    window.setMouseTracking(True)
    window.setWindowTitle('Livewire Demo')
    window.show()
    window.setFixedSize(window.size())
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()    
