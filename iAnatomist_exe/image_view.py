from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Signal
import numpy as np
from PySide6.QtCore import Qt
from PySide6 import QtCore


class ImageViewer(QtCore.QObject):
    ''' Basic image viewer class to show an image with zoom and pan functionaities.
        Requirement: Qt's Qlabel widget name where the image will be drawn/displayed.
    '''

    doubleclick = Signal(list)

    def __init__(self, qlabel):
        QtCore.QObject.__init__(self)
        self.qlabel_image = qlabel                            # widget/window name where image is displayed (I'm usiing qlabel)
        self.qimage_scaled = QImage()                         # scaled image to fit to the size of qlabel_image
        self.qpixmap = QPixmap()                              # qpixmap to fill the qlabel_image

        self.zoomX = 1              # zoom factor w.r.t size of qlabel_image
        self.position = [0, 0]      # position of top left corner of qimage_label w.r.t. qimage_scaled
        self.panFlag = True        # to enable or disable pan
        self.drawFlag = False

        self.drawX1 = 0
        self.drawY1 = 0
        self.drawX2 = 0
        self.drawY2 = 0


        self.qlabel_image.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.__connectEvents()

    def __connectEvents(self):
        # Mouse events
        self.qlabel_image.mousePressEvent = self.mousePressAction
        self.qlabel_image.mouseMoveEvent = self.mouseMoveAction
        self.qlabel_image.mouseReleaseEvent = self.mouseReleaseAction
        self.qlabel_image.wheelEvent = self.wheelScrollEvent

    def onResize(self):
        ''' things to do when qlabel_image is resized '''
        self.qpixmap = QPixmap(self.qlabel_image.size())
        self.qpixmap.fill(QtCore.Qt.gray)
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def loadImage(self, imagePath):
        ''' To load and display new image.'''
        # self.qimage = QImage(imagePath)
        imagePath = np.ascontiguousarray(imagePath)
        self.qimage = QImage(imagePath.data, imagePath.shape[1], imagePath.shape[0],imagePath.shape[1]*3, QImage.Format_RGB888)
        self.qpixmap = QPixmap(self.qlabel_image.size())
        if not self.qimage.isNull():
            # reset Zoom factor and Pan position
            # self.zoomX = 1
            # self.position = [0, 0]
            # self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width(), self.qlabel_image.height(), QtCore.Qt.KeepAspectRatio)
            self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX,
                                                    self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
            self.update()
        else:
            self.statusbar.showMessage('Cannot open this image! Try another one.', 5000)

    def update(self):
        ''' This function actually draws the scaled image to the qlabel_image.
            It will be repeatedly called when zooming or panning.
            So, I tried to include only the necessary operations required just for these tasks.
        '''
        if not self.qimage_scaled.isNull():
            # check if position is within limits to prevent unbounded panning.
            px, py = self.position
            px = px if (px <= self.qimage_scaled.width() - self.qlabel_image.width()) else (self.qimage_scaled.width() - self.qlabel_image.width())
            py = py if (py <= self.qimage_scaled.height() - self.qlabel_image.height()) else (self.qimage_scaled.height() - self.qlabel_image.height())
            px = px if (px >= 0) else 0
            py = py if (py >= 0) else 0
            self.position = (px, py)

            self.qpixmap.fill(QtCore.Qt.white)

            # the act of painting the qpixamp
            if self.panFlag:
                painter = QPainter()
                painter.begin(self.qpixmap)
                painter.drawImage(QtCore.QPoint(0, 0), self.qimage_scaled,
                        QtCore.QRect(self.position[0], self.position[1], self.qlabel_image.width(), self.qlabel_image.height()) )
                painter.end()

                self.qlabel_image.setPixmap(self.qpixmap)
            if self.drawFlag:
                painter = QPainter()
                painter.begin(self.qpixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                painter.drawImage(QtCore.QPoint(0, 0), self.qimage_scaled,
                                  QtCore.QRect(self.position[0], self.position[1], self.qlabel_image.width(),
                                               self.qlabel_image.height()))
                painter.setPen(QtGui.QPen(QtCore.Qt.yellow, 3))
                painter.drawLine(QtCore.QLine(self.drawX1, self.drawY1, self.drawX2, self.drawY2))
                painter.end()
                self.qlabel_image.setPixmap(self.qpixmap)
        else:
            pass

    def mousePressAction(self, QMouseEvent):
        if QMouseEvent.button() == Qt.LeftButton:
            self.clearDraw()
            x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
            self.drawX1 = x
            self.drawY1 = y
            #print(x,y)
            if self.panFlag or self.drawFlag:
                self.pressed = QMouseEvent.pos()    # starting point of drag vector
                self.anchor = self.position         # save the pan position when panning starts=
        else:
            x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
            scale = self.qimage_scaled.height() / self.qimage.height()
            x = int((x + self.position[0]) / scale)
            y = int((y + self.position[1]) / scale)
            self.doubleclick.emit([x, y])

    def mouseMoveAction(self, QMouseEvent):
        x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
        self.drawX2 = x
        self.drawY2 = y
        if self.pressed:
            dx, dy = x - self.pressed.x(), y - self.pressed.y()         # calculate the drag vector
            if self.panFlag:
                self.position = self.anchor[0] - dx, self.anchor[1] - dy    # update pan position using drag vector
            self.update()                                               # show the image with udated pan position

    def mouseReleaseAction(self, QMouseEvent):
        self.pressed = None                                             # clear the starting point of drag vector
        x, y = QMouseEvent.pos().x(), QMouseEvent.pos().y()
        self.drawX2 = x
        self.drawY2 = y


    def wheelScrollEvent(self, QWheelEvent):
        if QWheelEvent.angleDelta().y() > 0:
            self.zoomPlus()
        else:
            self.zoomMinus()

    def zoomPlus(self):
        self.zoomX += 1
        px, py = self.position
        px += self.qlabel_image.width()/2
        py += self.qlabel_image.height()/2
        self.position = (px, py)
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def zoomMinus(self):
        if self.zoomX > 1:
            self.zoomX -= 1
            px, py = self.position
            px -= self.qlabel_image.width()/2
            py -= self.qlabel_image.height()/2
            self.position = (px, py)
            self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
            self.update()

    def resetZoom(self):
        self.zoomX = 1
        self.position = [0, 0]
        self.qimage_scaled = self.qimage.scaled(self.qlabel_image.width() * self.zoomX, self.qlabel_image.height() * self.zoomX, QtCore.Qt.KeepAspectRatio)
        self.update()

    def enablePan(self, value):
        self.panFlag = value

    def enbaleDraw(self,value):
        self.drawFlag = value

    def getDrawInfo(self, z=0):
        scale = self.qimage_scaled.height()/self.qimage.height()
        drawX1 = int((self.drawX1 + self.position[0]) / scale)
        drawY1 = int((self.drawY1 + self.position[1]) / scale)
        drawX2 = int((self.drawX2 + self.position[0]) / scale)
        drawY2 = int((self.drawY2 + self.position[1]) / scale)
        origin_x = (drawX1 + drawX2)//2
        origin_y = (drawY1 + drawY2) // 2
        angle = ((drawY2 - drawY1) + 1e-6) / ((drawX2 - drawX1) + 1e-6)
        if z==0:
            if (drawY2 - drawY1) >= 0 and (drawX2 - drawX1) >= 0:
                angle = np.arctan(angle)
            elif (drawY2 - drawY1) >= 0 and (drawX2 - drawX1) < 0:
                angle = np.arctan(angle) + np.pi
            elif (drawY2 - drawY1) < 0 and (drawX2 - drawX1) <= 0:
                angle = np.arctan(angle) + np.pi
            elif (drawY2 - drawY1) < 0 and (drawX2 - drawX1) > 0:
                angle = np.arctan(angle) + np.pi*2
        if z==2:
            if (drawY2 - drawY1) >= 0 and (drawX2 - drawX1) >= 0:
                angle = np.arctan(angle)
            elif (drawY2 - drawY1) >= 0 and (drawX2 - drawX1) < 0:
                angle = np.arctan(angle) + np.pi
            elif (drawY2 - drawY1) < 0 and (drawX2 - drawX1) <= 0:
                angle = np.arctan(angle) + np.pi
            elif (drawY2 - drawY1) < 0 and (drawX2 - drawX1) > 0:
                angle = np.arctan(angle) + np.pi*2
        length = np.sqrt(np.square(drawY2 - drawY1) + np.square(drawX2 - drawX1))
        return origin_x, origin_y, angle, length

    def clearDraw(self):
        self.drawX1 = 0
        self.drawX2 = 0
        self.drawY1 = 0
        self.drawY2 = 0
