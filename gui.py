"""
A crappy GUI for intelligent scissors/livewire with 
# Left click to set seed
# Right click to finish
# Escape to export image file
"""

from __future__ import division
import time
import cv2
from PyQt4 import QtGui, QtCore
from threading import Thread
from livewire import Livewire

class ImageWin(QtGui.QWidget):
    def __init__(self):
        super(ImageWin, self).__init__()
        self.setupUi()
        self.active = False
        self.seed_enabled = True
        self.seed = None
        self.path_map = {}
        self.path = []
        
    def setupUi(self):
        self.hbox = QtGui.QVBoxLayout(self)
        
        # Load and initialize image and boundary map
        self.image_path = ''
        self.boundary_map_path = ''
        while self.image_path == '':
            self.image_path = QtGui.QFileDialog.getOpenFileName(self, '', '', '(*.bmp *.jpg *.png)')
        self.image = QtGui.QPixmap(self.image_path)
        self.cv2_image = cv2.imread(str(self.image_path))
        while self.boundary_map_path == '':
            self.boundary_map_path = QtGui.QFileDialog.getOpenFileName(self, '', '', '(*.bmp *.jpg *.png)')
        self.cv2_boundary_map = cv2.imread(str(self.boundary_map_path))
        self.lw = Livewire(self.cv2_image, self.cv2_boundary_map)
        self.w, self.h = self.image.width(), self.image.height()
        
        self.canvas = QtGui.QLabel(self)
        self.canvas.setMouseTracking(True)
        self.canvas.setPixmap(self.image)
        
        self.status_bar = QtGui.QStatusBar(self)
        self.status_bar.showMessage('Left click to set a seed')
        
        self.hbox.addWidget(self.canvas)
        self.hbox.addWidget(self.status_bar)
        self.setLayout(self.hbox)
   
    def keyPressEvent(self, event):
            pos = QtGui.QCursor.pos()
            x, y = pos.x()-self.canvas.x(), pos.y()-self.canvas.y()
            
            if x < 0:
                x = 0
            if x >= self.w:
                x = self.w - 1
            if y < 0:
                y = 0
            if y >= self.h:
                y = self.h - 1

            # Get the mouse cursor position
            p = y, x
            seed = self.seed
            
            # Export bitmap
            if event.key() == QtCore.Qt.Key_Escape:
                filepath = QtGui.QFileDialog.getSaveFileName(self, 'Save image audio to', '', '*.jpg\n')
                if not filepath.endsWith('.jpg'):
                    filepath += '.jpg'
                
                image = QtGui.QPixmap(self.image.size())
                
                image.fill(QtCore.Qt.black)
 
                draw = QtGui.QPainter()
                draw.begin(image)
                draw.setPen(QtCore.Qt.white)
                if self.path:
                    draw.setPen(QtCore.Qt.white)
                    for p in self.path:
                        draw.drawPoint(p[1], p[0])
                draw.end()
                
                image.save(filepath, quality=100)
 
    def mousePressEvent(self, event):            
            if event.buttons() == QtCore.Qt.LeftButton:
                pos = event.pos()
                x, y = pos.x()-self.canvas.x(), pos.y()-self.canvas.y()
            
                if x < 0:
                    x = 0
                if x >= self.w:
                    x = self.w - 1
                if y < 0:
                    y = 0
                if y >= self.h:
                    y = self.h - 1

                # Get the mouse cursor position
                p = y, x
                seed = self.seed
            
                self.seed = p
                self.seed_enabled = True
                
                if seed is not None and self.path_map:
                    while p != seed:
                        p = self.path_map[p]
                        self.path.append(p)
              
            # Finish current task and reset
            elif event.buttons() == QtCore.Qt.RightButton:
                self.path_map = {}
                self.seed_enabled = False
                self.seed = None
                self.status_bar.showMessage('Left click to set a seed')
                self.active = False
    
    def mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.NoButton:
            pos = event.pos()
            x, y = pos.x()-self.canvas.x(), pos.y()-self.canvas.y()

            if x < 0 or x >= self.w or y < 0 or y >= self.h:
                pass
            else:
                p = y, x
                path = []
                
                # Calculate path map
                if self.seed_enabled and self.seed is not None:
                    #Thread(target=self._update_path_map_progress).start()
                    self._cal_path_matrix(p)
                
                    # Draw livewire
                    while p != self.seed:
                        p = self.path_map[p]
                        path.append(p)
                
                image = self.image.copy()
                draw = QtGui.QPainter()
                draw.begin(image)
                draw.setPen(QtCore.Qt.blue)
                for p in path:
                    draw.drawPoint(p[1], p[0])
                if self.path:
                    draw.setPen(QtCore.Qt.green)
                    for p in self.path:
                        draw.drawPoint(p[1], p[0])
                draw.end()
                self.canvas.setPixmap(image)
    
    def _cal_path_matrix(self, p):
        self.seed_enabled = False
        self.active = False
        #self.status_bar.showMessage('Calculating path map...')
        path_matrix = self.lw.get_path_matrix(self.seed, p)
        #self.status_bar.showMessage(r'Left: new seed / Right: finish')
        self.seed_enabled = True
        self.active = True
        
        self.path_map = path_matrix
    
    def _update_path_map_progress(self):
        while not self.seed_enabled:
            time.sleep(0.1)
            message = 'Calculating path map... {:.1f}%'.format(self.lw.n_processed/self.lw.n_pixs*100.0)
            self.status_bar.showMessage(message)
        self.status_bar.showMessage(r'Left: new seed / Right: finish')
