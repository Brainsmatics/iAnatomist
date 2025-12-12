#!/usr/bin/env python

''' A basic GUi to use ImageViewer class to show its functionalities and use cases. '''
import multiprocessing
import shutil
import time

from PySide6 import QtCore, QtWidgets, QtWidgets
from image_view import ImageViewer
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt, QThread, Signal, Slot
import sys, os
import numpy as np
from mainUi import Ui_MainWindow
import cv2
from skimage import transform
import pandas as pd
import SimpleITK as sitk
from utils import *
from affine_registration import affineReg
from deform_registration import deformReg
from subregion_registration import subReg
import PySide6


dirname = os.path.dirname(PySide6.__file__)
plugin_path = os.path.join(dirname, 'plugins', 'platforms')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path


class Iwindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.image_viewer3d_1 = ImageViewer(self.imLabelX)
        self.image_viewer3d_2 = ImageViewer(self.imLabelY)
        self.image_viewer3d_3 = ImageViewer(self.imLabelZ)

        self.reg3Scroll_x = 0
        self.reg3Scroll_y = 0
        self.reg3Scroll_z = 0

        self.modify_xysize = self.spinBox_xy.value()
        self.modify_zsize = self.spinBox_z.value()

        self.mode = "img"
        self.draw = False
        self.transform = None
        self.dst_path = None
        self.affine_matrics = None
        self.image = None
        self.flow_high = None

        self.ann3Img = sitk.GetArrayFromImage(sitk.ReadImage('./cache/annotation_25.nii.gz'))
        self.ann3EdgeImg = sitk.GetArrayFromImage(sitk.GradientMagnitude(sitk.GetImageFromArray(self.ann3Img)))
        self.shape = self.ann3Img.shape

        self.createMenusActions()
        self.__connectEvents()
        self.showMaximized()

    def createMenusActions(self):
        # add menu actions
        self.actionOpen = QAction("Open image", self, shortcut="Ctrl+O",
                               triggered=self.open_image)
        self.actionSave_warped_image = QAction("&Save warped image", self, shortcut="Ctrl+s",
                                    triggered=self.save_warped_image)
        self.actionSave_linear_transform = QAction("&Save Save_linear_transform", self, shortcut="Ctrl+s",
                                         triggered=self.save_linear_transform)
        self.actionSave_nolinear_transform = QAction("&Save_nolinear_transform", self,
                                  shortcut="d", enabled=True, triggered=self.save_nolinear_transform)
        self.actionView = QAction("&View", self,
                                  shortcut="v", enabled=True, triggered=self.action_view)
        self.actionDraw = QAction("&Draw", self,
                                  shortcut="v", enabled=True, triggered=self.action_draw)
        self.actionUndo = QAction("&Undo", self,
                                  shortcut="Ctrl+z", enabled=True, triggered=self.undo)

        # add menu
        self.menuFile.addAction(self.actionOpen)
        # self.menuSave.addAction(self.saveAct)
        self.menuSave = QtWidgets.QMenu("&Save", self)
        self.menuSave.addAction(self.actionSave_warped_image)
        self.menuSave.addAction(self.actionSave_linear_transform)

        self.menuFile.addMenu(self.menuSave)
        self.menuMode.addAction(self.actionView)
        self.menuMode.addAction(self.actionDraw)
        self.menuMode.addAction(self.actionUndo)

    def __connectEvents(self):
        # transform modify
        self.pushButton_x.clicked.connect(self.updata_transform_x)
        self.pushButton_y.clicked.connect(self.updata_transform_y)
        self.pushButton_z.clicked.connect(self.updata_transform_z)
        # apply registration
        self.pushButton_applyReg.clicked.connect(self.registration)
        # # apply registration(sub)
        # self.pushButton_applyRegSub.clicked.connect(self.registrationSub)
        # value change of ScrollBar
        self.horizontalScrollBar_x.valueChanged.connect(self.reg3Scroll_x_change)
        self.horizontalScrollBar_y.valueChanged.connect(self.reg3Scroll_y_change)
        self.horizontalScrollBar_z.valueChanged.connect(self.reg3Scroll_z_change)
        self.image_viewer3d_1.doubleclick.connect(self.reg3Locate1)
        self.image_viewer3d_2.doubleclick.connect(self.reg3Locate2)
        self.image_viewer3d_3.doubleclick.connect(self.reg3Locate3)

    @Slot()
    def action_draw(self):
        self.imLabelX.setCursor(QtCore.Qt.CrossCursor)
        self.imLabelY.setCursor(QtCore.Qt.CrossCursor)
        self.imLabelZ.setCursor(QtCore.Qt.CrossCursor)
        self.image_viewer3d_1.enablePan(False)
        self.image_viewer3d_1.enbaleDraw(True)
        self.image_viewer3d_2.enablePan(False)
        self.image_viewer3d_2.enbaleDraw(True)
        self.image_viewer3d_3.enablePan(False)
        self.image_viewer3d_3.enbaleDraw(True)
        self.draw = True

    @Slot()
    def action_view(self):
        self.imLabelX.setCursor(QtCore.Qt.OpenHandCursor)
        self.imLabelY.setCursor(QtCore.Qt.OpenHandCursor)
        self.imLabelZ.setCursor(QtCore.Qt.OpenHandCursor)
        self.image_viewer3d_1.enablePan(True)
        self.image_viewer3d_1.enbaleDraw(False)
        self.image_viewer3d_2.enablePan(True)
        self.image_viewer3d_2.enbaleDraw(False)
        self.image_viewer3d_3.enablePan(True)
        self.image_viewer3d_3.enbaleDraw(False)
        self.draw = False

    @Slot()
    def open_image(self):
        fileName, filetype = QtWidgets.QFileDialog.getOpenFileName(self, "Open File",
                                                                   QtCore.QDir.currentPath())
        self.dst_path = os.path.dirname(fileName)
        self.image = sitk.GetArrayFromImage(sitk.ReadImage(fileName))
        self.transform = np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2], 3))
        self.updata_reg3ImageView()
        self.updata_reg3ScrollValue_range()

    @Slot()
    def save_warped_image(self):
        fileName, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "Save warped image",
                                                                   QtCore.QDir.currentPath(),
                                                                   "Image Files (*.nii.gz)")
        sitk.WriteImage(sitk.GetImageFromArray(self.image), fileName)

    @Slot()
    def save_linear_transform(self):
        fileName, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "Save warped image",
                                                                   QtCore.QDir.currentPath(),
                                                                   "Transform Files (*.mat)")
        shutil.copy(self.affine_matrics, fileName)


    @Slot()
    def save_nolinear_transform(self):
        fileName, filetype = QtWidgets.QFileDialog.getSaveFileName(self, "Save warped image",
                                                                   QtCore.QDir.currentPath(),
                                                                   "Transform Files (*.nii.gz)")
        # shutil.copy('./cache/deform/deform_transform.nii.gz', fileName)
        sitk.WriteImage(sitk.GetImageFromArray(self.flow_high), fileName)

    # 根据选定的线性或者非线性配准进行配准的选择和执行
    @Slot()
    def registration(self):
        # print(time.time())
        if self.checkBox_linear.isChecked():
            self.image, self.affine_matrics = affineReg(self.image)
        # print(time.time())
        if self.checkBox_nolinear.isChecked():
            self.image, self.flow_high= deformReg(self.image)
        # print(time.time())
        self.mode = 'ann'
        self.updata_reg3ImageView()

    # 已弃用
    # @Slot()
    # def registrationSub(self):
    #     if self.comboBox_sub.currentText() == 'TH':
    #         self.subImage = self.image[220: 362, 103:225, 110:345]
    #         self.subImage = subReg(self.subImage)
    #         self.image[220: 362, 103:225, 110:345] = self.subImage
    #     self.updata_reg3ImageView()

    @Slot()
    def undo(self):
        self.image = self.undo_image
        self.transform = self.undo_transform
        self.updata_reg3ImageView()

    @Slot()
    def reg3Scroll_x_change(self):
        self.reg3Scroll_x = self.horizontalScrollBar_x.value()
        self.updata_reg3ScrollValue_range()

    @Slot()
    def reg3Scroll_y_change(self):
        self.reg3Scroll_y = self.horizontalScrollBar_y.value()
        self.updata_reg3ScrollValue_range()

    @Slot()
    def reg3Scroll_z_change(self):
        self.reg3Scroll_z = self.horizontalScrollBar_z.value()
        self.updata_reg3ScrollValue_range()

    '''
        双击某一个面的图像中的某一个点时，另外两个面换到对应的这个点的坐标的面上，同时显示该点的坐标以及对应的脑区位置
    '''
    @Slot()
    def reg3Locate1(self, xy_list):
        if not self.draw:
            self.horizontalScrollBar_y.setValue(xy_list[1]-1)
            self.horizontalScrollBar_z.setValue(xy_list[0]-1)
            self.label_loc.setText("point loc: {}, {}, {}".format(self.reg3Scroll_z, self.reg3Scroll_y, self.reg3Scroll_x))
            index = dict(pd.read_csv("./new_correspoding_file.csv",header=None).iloc[:,:2])[1]
            self.label_area.setText("point aread: {}".format(
                index[self.ann3Img[self.reg3Scroll_x, self.reg3Scroll_y, self.reg3Scroll_z]]))
            self.updata_reg3ImageView()

    @Slot()
    def reg3Locate2(self, xy_list):
        if not self.draw:
            self.horizontalScrollBar_x.setValue(xy_list[1] - 1)
            self.horizontalScrollBar_z.setValue(xy_list[0] - 1)
            self.label_loc.setText(
                "point loc: {}, {}, {}".format(self.reg3Scroll_z, self.reg3Scroll_y, self.reg3Scroll_x))
            index = dict(pd.read_csv("./new_correspoding_file.csv", header=None).iloc[:, :2])[1]
            self.label_area.setText("point aread: {}".format(
                index[self.ann3Img[self.reg3Scroll_x, self.reg3Scroll_y, self.reg3Scroll_z]]))
            self.updata_reg3ImageView()

    @Slot()
    def reg3Locate3(self, xy_list):
        if not self.draw:
            self.horizontalScrollBar_x.setValue(xy_list[1] - 1)
            self.horizontalScrollBar_y.setValue(xy_list[0] - 1)
            self.label_loc.setText(
                "point loc: {}, {}, {}".format(self.reg3Scroll_z, self.reg3Scroll_y, self.reg3Scroll_x))
            index = dict(pd.read_csv("./new_correspoding_file.csv", header=None).iloc[:, :2])[1]
            self.label_area.setText("point aread: {}".format(
                index[self.ann3Img[self.reg3Scroll_x, self.reg3Scroll_y, self.reg3Scroll_z]]))
            self.updata_reg3ImageView()

    @Slot()
    def updata_reg3ScrollValue_range(self):
        self.horizontalScrollBar_x.setRange(0,self.image.shape[0]-1)
        self.horizontalScrollBar_y.setRange(0, self.image.shape[1]-1)
        self.horizontalScrollBar_z.setRange(0, self.image.shape[2]-1)
        self.updata_reg3ImageView()

    # 更新三个面的显示图像
    @Slot()
    def updata_reg3ImageView(self):
        if self.mode == "img":
            self.image_viewer3d_1.loadImage(
                cv2.cvtColor(cv2.equalizeHist(self.image[self.reg3Scroll_x, :, :].astype(np.uint8)), cv2.COLOR_GRAY2RGB))
            self.image_viewer3d_2.loadImage(
                cv2.cvtColor(cv2.equalizeHist(self.image[:, self.reg3Scroll_y, :].astype(np.uint8)).transpose(1,0), cv2.COLOR_GRAY2RGB))
            self.image_viewer3d_3.loadImage(
                cv2.cvtColor(cv2.equalizeHist(self.image[:, :, self.reg3Scroll_z].astype(np.uint8)).transpose(1,0), cv2.COLOR_GRAY2RGB))
        elif self.mode == "ann":
            img_x = cv2.cvtColor(cv2.equalizeHist(
                self.image[self.reg3Scroll_x, :, :].astype(np.uint8)), cv2.COLOR_GRAY2RGB)
            img_y = cv2.cvtColor(cv2.equalizeHist(
                self.image[:, self.reg3Scroll_y, :].astype(np.uint8)), cv2.COLOR_GRAY2RGB)
            img_z = cv2.cvtColor(cv2.equalizeHist(
                self.image[:, :, self.reg3Scroll_z].astype(np.uint8)), cv2.COLOR_GRAY2RGB)
            if self.image.shape != self.ann3EdgeImg.shape:
                self.ann3EdgeImg = transform.resize(self.ann3EdgeImg, output_shape=self.image.shape,
                                                    order=0, anti_aliasing=False, preserve_range=True)
            ann_x = self.ann3EdgeImg[self.reg3Scroll_x, :, :].astype(np.uint8)
            ann_y = self.ann3EdgeImg[:, self.reg3Scroll_y, :].astype(np.uint8)
            ann_z = self.ann3EdgeImg[:, :, self.reg3Scroll_z].astype(np.uint8)
            ann_x[ann_x > 0] = 1
            ann_y[ann_y > 0] = 1
            ann_z[ann_z > 0] = 1
            ann_x = np.stack([ann_x * 200, ann_x * 1, ann_x * 1], axis=2)
            ann_y = np.stack([ann_y * 200, ann_y * 1, ann_y * 1], axis=2)
            ann_z = np.stack([ann_z * 200, ann_z * 1, ann_z * 1], axis=2)
            # img_x = img_x + 0.9 * np.stack([ann_x * 255, ann_x * 1, ann_x * 1], axis=2)
            # img_y = img_y + 0.9 * np.stack([ann_y * 255, ann_y * 1, ann_y * 1], axis=2)
            # img_z = img_z + 0.9 * np.stack([ann_z * 255, ann_z * 1, ann_z * 1], axis=2)
            img_x[ann_x != 0] = ann_x[ann_x!=0]
            img_y[ann_y != 0] = ann_y[ann_y!=0]
            img_z[ann_z != 0] = ann_z[ann_z!=0]
            # img_x[img_x > 255] = 255
            # img_y[img_y > 255] = 255
            # img_z[img_z > 255] = 255
            self.image_viewer3d_1.loadImage(img_x.astype(np.uint8))
            self.image_viewer3d_2.loadImage(img_y.astype(np.uint8).transpose(1,0,2))
            self.image_viewer3d_3.loadImage(img_z.astype(np.uint8).transpose(1,0,2))

    # 对于冠状面修改的应用函数
    @Slot()
    def updata_transform_x(self):
        self.mode = 'ann'
        origin_x, origin_y, angle, length = self.image_viewer3d_1.getDrawInfo()
        modify_xysize = self.spinBox_xy.value()
        modify_zsize = self.spinBox_z.value()
        self.undo_image = self.image
        self.undo_transform = self.transform
        self.image, self.transform = transform3d_modify(self.image, self.transform, 0,
                                                          (self.reg3Scroll_x, origin_y, origin_x),
                                                          angle, length / 10, modify_xysize)
        self.image_viewer3d_1.clearDraw()
        self.updata_reg3ImageView()

    # 对于矢状面修改的应用函数
    @Slot()
    def updata_transform_y(self):
        self.mode = 'ann'
        origin_x, origin_y, angle, length = self.image_viewer3d_2.getDrawInfo()
        modify_xysize = self.spinBox_xy.value()
        modify_zsize = self.spinBox_z.value()
        self.undo_image = self.image
        self.undo_transform = self.transform
        self.image, self.transform = transform3d_modify(self.image, self.transform, 1,
                                                          (origin_x, self.reg3Scroll_y, origin_y), angle, length / 10,
                                                          modify_xysize)
        self.image_viewer3d_2.clearDraw()
        self.updata_reg3ImageView()

    # 对于水平面修改的应用函数
    @Slot()
    def updata_transform_z(self):
        self.mode = 'ann'
        origin_x, origin_y, angle, length = self.image_viewer3d_3.getDrawInfo(z=2)
        modify_xysize = self.spinBox_xy.value()
        modify_zsize = self.spinBox_z.value()
        self.undo_image = self.image
        self.undo_transform = self.transform
        self.image, self.transform = transform3d_modify(self.image, self.transform, 2,
                                                          (origin_x, origin_y, self.reg3Scroll_z), angle, length / 10,
                                                          modify_xysize)
        self.image_viewer3d_3.clearDraw()
        self.updata_reg3ImageView()


def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle(QtWidgets.QStyleFactory.create("Cleanlooks"))
    app.setPalette(QtWidgets.QApplication.style().standardPalette())
    app.setStyle("fusion")
    parentWindow = Iwindow(None)
    sys.exit(app.exec_())


if __name__ == "__main__":
    # print __doc__
    multiprocessing.freeze_support()
    main()

