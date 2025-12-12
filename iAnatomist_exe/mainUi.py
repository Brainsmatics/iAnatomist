# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainUi.ui'
##
## Created by: Qt User Interface Compiler version 6.4.0
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLayout,
    QMainWindow, QMenu, QMenuBar, QPushButton,
    QScrollBar, QSizePolicy, QSpinBox, QStatusBar,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1113, 761)
        self.actionOpen = QAction(MainWindow)
        self.actionOpen.setObjectName(u"actionOpen")
        self.actionSave_warped_image = QAction(MainWindow)
        self.actionSave_warped_image.setObjectName(u"actionSave_warped_image")
        self.actionSave_linear_transform = QAction(MainWindow)
        self.actionSave_linear_transform.setObjectName(u"actionSave_linear_transform")
        self.actionSave_nolinear_transform = QAction(MainWindow)
        self.actionSave_nolinear_transform.setObjectName(u"actionSave_nolinear_transform")
        self.actionView = QAction(MainWindow)
        self.actionView.setObjectName(u"actionView")
        self.actionDraw = QAction(MainWindow)
        self.actionDraw.setObjectName(u"actionDraw")
        self.actionUndo = QAction(MainWindow)
        self.actionUndo.setObjectName(u"actionUndo")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.horizontalLayout_21 = QHBoxLayout(self.centralwidget)
        self.horizontalLayout_21.setObjectName(u"horizontalLayout_21")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.verticalLayout_x = QVBoxLayout()
        self.verticalLayout_x.setObjectName(u"verticalLayout_x")
        self.verticalLayout_x.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.imLabelX = QLabel(self.centralwidget)
        self.imLabelX.setObjectName(u"imLabelX")
        self.imLabelX.setAlignment(Qt.AlignCenter)

        self.verticalLayout_x.addWidget(self.imLabelX)

        self.horizontalScrollBar_x = QScrollBar(self.centralwidget)
        self.horizontalScrollBar_x.setObjectName(u"horizontalScrollBar_x")
        self.horizontalScrollBar_x.setOrientation(Qt.Horizontal)

        self.verticalLayout_x.addWidget(self.horizontalScrollBar_x)

        self.pushButton_x = QPushButton(self.centralwidget)
        self.pushButton_x.setObjectName(u"pushButton_x")
        font = QFont()
        font.setPointSize(10)
        self.pushButton_x.setFont(font)

        self.verticalLayout_x.addWidget(self.pushButton_x)


        self.gridLayout.addLayout(self.verticalLayout_x, 0, 1, 1, 1)

        self.verticalLayout_z = QVBoxLayout()
        self.verticalLayout_z.setObjectName(u"verticalLayout_z")
        self.verticalLayout_z.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.imLabelZ = QLabel(self.centralwidget)
        self.imLabelZ.setObjectName(u"imLabelZ")
        self.imLabelZ.setAlignment(Qt.AlignCenter)

        self.verticalLayout_z.addWidget(self.imLabelZ)

        self.horizontalScrollBar_z = QScrollBar(self.centralwidget)
        self.horizontalScrollBar_z.setObjectName(u"horizontalScrollBar_z")
        self.horizontalScrollBar_z.setOrientation(Qt.Horizontal)

        self.verticalLayout_z.addWidget(self.horizontalScrollBar_z)

        self.pushButton_z = QPushButton(self.centralwidget)
        self.pushButton_z.setObjectName(u"pushButton_z")
        self.pushButton_z.setFont(font)

        self.verticalLayout_z.addWidget(self.pushButton_z)


        self.gridLayout.addLayout(self.verticalLayout_z, 1, 1, 1, 1)

        self.verticalLayout_y = QVBoxLayout()
        self.verticalLayout_y.setObjectName(u"verticalLayout_y")
        self.verticalLayout_y.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.imLabelY = QLabel(self.centralwidget)
        self.imLabelY.setObjectName(u"imLabelY")
        self.imLabelY.setAlignment(Qt.AlignCenter)

        self.verticalLayout_y.addWidget(self.imLabelY)

        self.horizontalScrollBar_y = QScrollBar(self.centralwidget)
        self.horizontalScrollBar_y.setObjectName(u"horizontalScrollBar_y")
        self.horizontalScrollBar_y.setOrientation(Qt.Horizontal)

        self.verticalLayout_y.addWidget(self.horizontalScrollBar_y)

        self.pushButton_y = QPushButton(self.centralwidget)
        self.pushButton_y.setObjectName(u"pushButton_y")
        self.pushButton_y.setFont(font)

        self.verticalLayout_y.addWidget(self.pushButton_y)


        self.gridLayout.addLayout(self.verticalLayout_y, 1, 0, 1, 1)

        self.groupBox_settings = QGroupBox(self.centralwidget)
        self.groupBox_settings.setObjectName(u"groupBox_settings")
        sizePolicy = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_settings.sizePolicy().hasHeightForWidth())
        self.groupBox_settings.setSizePolicy(sizePolicy)
        self.groupBox_settings.setMaximumSize(QSize(160000, 16777215))
        font1 = QFont()
        font1.setPointSize(12)
        font1.setBold(False)
        self.groupBox_settings.setFont(font1)
        self.horizontalLayout_19 = QHBoxLayout(self.groupBox_settings)
        self.horizontalLayout_19.setObjectName(u"horizontalLayout_19")
        self.verticalLayout_11 = QVBoxLayout()
        self.verticalLayout_11.setObjectName(u"verticalLayout_11")
        self.horizontalLayout_18 = QHBoxLayout()
        self.horizontalLayout_18.setObjectName(u"horizontalLayout_18")
        self.groupBox_registration = QGroupBox(self.groupBox_settings)
        self.groupBox_registration.setObjectName(u"groupBox_registration")
        sizePolicy1 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.groupBox_registration.sizePolicy().hasHeightForWidth())
        self.groupBox_registration.setSizePolicy(sizePolicy1)
        self.horizontalLayout_20 = QHBoxLayout(self.groupBox_registration)
        self.horizontalLayout_20.setObjectName(u"horizontalLayout_20")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.checkBox_linear = QCheckBox(self.groupBox_registration)
        self.checkBox_linear.setObjectName(u"checkBox_linear")

        self.horizontalLayout_10.addWidget(self.checkBox_linear)

        self.checkBox_nolinear = QCheckBox(self.groupBox_registration)
        self.checkBox_nolinear.setObjectName(u"checkBox_nolinear")

        self.horizontalLayout_10.addWidget(self.checkBox_nolinear)


        self.verticalLayout_4.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.label_none = QLabel(self.groupBox_registration)
        self.label_none.setObjectName(u"label_none")
        sizePolicy2 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label_none.sizePolicy().hasHeightForWidth())
        self.label_none.setSizePolicy(sizePolicy2)

        self.horizontalLayout_11.addWidget(self.label_none)

        self.pushButton_applyReg = QPushButton(self.groupBox_registration)
        self.pushButton_applyReg.setObjectName(u"pushButton_applyReg")
        font2 = QFont()
        font2.setPointSize(10)
        font2.setBold(False)
        self.pushButton_applyReg.setFont(font2)

        self.horizontalLayout_11.addWidget(self.pushButton_applyReg)


        self.verticalLayout_4.addLayout(self.horizontalLayout_11)


        self.horizontalLayout_20.addLayout(self.verticalLayout_4)


        self.horizontalLayout_18.addWidget(self.groupBox_registration)

        self.groupBox_registrationSub = QGroupBox(self.groupBox_settings)
        self.groupBox_registrationSub.setObjectName(u"groupBox_registrationSub")
        sizePolicy1.setHeightForWidth(self.groupBox_registrationSub.sizePolicy().hasHeightForWidth())
        self.groupBox_registrationSub.setSizePolicy(sizePolicy1)
        self.horizontalLayout_15 = QHBoxLayout(self.groupBox_registrationSub)
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.verticalLayout_9 = QVBoxLayout()
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.label_subregion = QLabel(self.groupBox_registrationSub)
        self.label_subregion.setObjectName(u"label_subregion")
        sizePolicy1.setHeightForWidth(self.label_subregion.sizePolicy().hasHeightForWidth())
        self.label_subregion.setSizePolicy(sizePolicy1)
        self.label_subregion.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_16.addWidget(self.label_subregion)

        self.comboBox_sub = QComboBox(self.groupBox_registrationSub)
        self.comboBox_sub.addItem("")
        self.comboBox_sub.setObjectName(u"comboBox_sub")

        self.horizontalLayout_16.addWidget(self.comboBox_sub)


        self.verticalLayout_9.addLayout(self.horizontalLayout_16)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.label_none_3 = QLabel(self.groupBox_registrationSub)
        self.label_none_3.setObjectName(u"label_none_3")
        sizePolicy2.setHeightForWidth(self.label_none_3.sizePolicy().hasHeightForWidth())
        self.label_none_3.setSizePolicy(sizePolicy2)

        self.horizontalLayout_17.addWidget(self.label_none_3)

        self.pushButton_applyRegSub = QPushButton(self.groupBox_registrationSub)
        self.pushButton_applyRegSub.setObjectName(u"pushButton_applyRegSub")
        self.pushButton_applyRegSub.setFont(font2)

        self.horizontalLayout_17.addWidget(self.pushButton_applyRegSub)


        self.verticalLayout_9.addLayout(self.horizontalLayout_17)


        self.horizontalLayout_15.addLayout(self.verticalLayout_9)


        self.horizontalLayout_18.addWidget(self.groupBox_registrationSub)


        self.verticalLayout_11.addLayout(self.horizontalLayout_18)

        self.groupBox_modify = QGroupBox(self.groupBox_settings)
        self.groupBox_modify.setObjectName(u"groupBox_modify")
        self.horizontalLayout_25 = QHBoxLayout(self.groupBox_modify)
        self.horizontalLayout_25.setObjectName(u"horizontalLayout_25")
        self.horizontalLayout_24 = QHBoxLayout()
        self.horizontalLayout_24.setSpacing(20)
        self.horizontalLayout_24.setObjectName(u"horizontalLayout_24")
        self.verticalLayout_10 = QVBoxLayout()
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.verticalLayout_10.setContentsMargins(20, -1, -1, -1)
        self.label_area = QLabel(self.groupBox_modify)
        self.label_area.setObjectName(u"label_area")

        self.verticalLayout_10.addWidget(self.label_area)

        self.label_loc = QLabel(self.groupBox_modify)
        self.label_loc.setObjectName(u"label_loc")

        self.verticalLayout_10.addWidget(self.label_loc)


        self.horizontalLayout_24.addLayout(self.verticalLayout_10)

        self.verticalLayout_13 = QVBoxLayout()
        self.verticalLayout_13.setObjectName(u"verticalLayout_13")
        self.verticalLayout_13.setContentsMargins(100, -1, 20, -1)
        self.horizontalLayout_22 = QHBoxLayout()
        self.horizontalLayout_22.setObjectName(u"horizontalLayout_22")
        self.label_5 = QLabel(self.groupBox_modify)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_22.addWidget(self.label_5)

        self.spinBox_xy = QSpinBox(self.groupBox_modify)
        self.spinBox_xy.setObjectName(u"spinBox_xy")
        self.spinBox_xy.setValue(4)

        self.horizontalLayout_22.addWidget(self.spinBox_xy)


        self.verticalLayout_13.addLayout(self.horizontalLayout_22)

        self.horizontalLayout_23 = QHBoxLayout()
        self.horizontalLayout_23.setObjectName(u"horizontalLayout_23")
        self.label_8 = QLabel(self.groupBox_modify)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setAlignment(Qt.AlignCenter)

        self.horizontalLayout_23.addWidget(self.label_8)

        self.spinBox_z = QSpinBox(self.groupBox_modify)
        self.spinBox_z.setObjectName(u"spinBox_z")
        self.spinBox_z.setValue(2)

        self.horizontalLayout_23.addWidget(self.spinBox_z)


        self.verticalLayout_13.addLayout(self.horizontalLayout_23)


        self.horizontalLayout_24.addLayout(self.verticalLayout_13)


        self.horizontalLayout_25.addLayout(self.horizontalLayout_24)


        self.verticalLayout_11.addWidget(self.groupBox_modify)


        self.horizontalLayout_19.addLayout(self.verticalLayout_11)


        self.gridLayout.addWidget(self.groupBox_settings, 0, 0, 1, 1)


        self.horizontalLayout_21.addLayout(self.gridLayout)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1113, 22))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuMode = QMenu(self.menubar)
        self.menuMode.setObjectName(u"menuMode")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuMode.menuAction())

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.actionOpen.setText(QCoreApplication.translate("MainWindow", u"Open image", None))
        self.actionSave_warped_image.setText(QCoreApplication.translate("MainWindow", u"Save warped image", None))
        self.actionSave_linear_transform.setText(QCoreApplication.translate("MainWindow", u"Save linear transform", None))
        self.actionSave_nolinear_transform.setText(QCoreApplication.translate("MainWindow", u"Save nolinear transform", None))
        self.actionView.setText(QCoreApplication.translate("MainWindow", u"View", None))
        self.actionDraw.setText(QCoreApplication.translate("MainWindow", u"Draw", None))
        self.actionUndo.setText(QCoreApplication.translate("MainWindow", u"Undo", None))
        self.imLabelX.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.pushButton_x.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.imLabelZ.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.pushButton_z.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.imLabelY.setText(QCoreApplication.translate("MainWindow", u"TextLabel", None))
        self.pushButton_y.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.groupBox_settings.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.groupBox_registration.setTitle(QCoreApplication.translate("MainWindow", u"Registration", None))
        self.checkBox_linear.setText(QCoreApplication.translate("MainWindow", u"Linear ", None))
        self.checkBox_nolinear.setText(QCoreApplication.translate("MainWindow", u"Nolinear", None))
        self.label_none.setText("")
        self.pushButton_applyReg.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.groupBox_registrationSub.setTitle(QCoreApplication.translate("MainWindow", u"Registration (subregion)", None))
        self.label_subregion.setText(QCoreApplication.translate("MainWindow", u"Subregion", None))
        self.comboBox_sub.setItemText(0, QCoreApplication.translate("MainWindow", u"TH", None))

        self.label_none_3.setText("")
        self.pushButton_applyRegSub.setText(QCoreApplication.translate("MainWindow", u"Apply", None))
        self.groupBox_modify.setTitle(QCoreApplication.translate("MainWindow", u"Modify and Info", None))
        self.label_area.setText(QCoreApplication.translate("MainWindow", u"Point Area: ", None))
        self.label_loc.setText(QCoreApplication.translate("MainWindow", u"Point Location:", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"xy:", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"z:", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuMode.setTitle(QCoreApplication.translate("MainWindow", u"Mode", None))
    # retranslateUi

