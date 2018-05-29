# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------

"""
This example demonstrates isocurve for triangular mesh with vertex data and a
 qt interface.
"""

import sys
import numpy as np
import os.path as osp
from vispy import scene
from vispy.scene import visuals
import cv2
try:
    from sip import setapi
    setapi("QVariant", 2)
    setapi("QString", 2)
except ImportError:
    pass
from PyQt4 import QtGui, QtCore

PARAMETERS = [('rpn_cnt', 1, 100, 'int', 50),
             ]

class Parameters(object):
    def __init__(self,parameters):
        self.parameters = parameters
        self.props = dict()
        self.props['rpn_visible'] = False
        self.props['method'] = 'KITTI'
        for nameV, minV, maxV, typeV, iniV in parameters:
            self.props[nameV] = iniV

class ObjectWidget(QtGui.QWidget):
    """
    Widget for editing OBJECT parameters
    """
    signal_objet_changed = QtCore.pyqtSignal(name='objectChanged')

    def __init__(self, parent=None):
        super(ObjectWidget, self).__init__(parent)
        self.param = Parameters(PARAMETERS)

        l_nbr_steps = QtGui.QLabel("Show rpn cnt")
        self.nbr_steps = QtGui.QSpinBox()
        self.nbr_steps.setMinimum(self.param.parameters[0][1])
        self.nbr_steps.setMaximum(self.param.parameters[0][2])
        self.nbr_steps.setValue(self.param.props['rpn_cnt'])
        self.nbr_steps.valueChanged.connect(self.update_param)

        l_cmap = QtGui.QLabel("Dataset ")
        self.cmap = ['KITTI', 'STi']
        self.combo = QtGui.QComboBox(self)
        self.combo.addItems(self.cmap)
        self.combo.setCurrentIndex(self.cmap.index(self.param.props['method']))
        self.combo.currentIndexChanged.connect(self.update_param)

        # Checkbox for whether or not the pivot point is visible
        l_rpn_chk = QtGui.QLabel("RPN boxes ")
        self.rpn_chk = QtGui.QCheckBox(self)
        self.rpn_chk.setChecked(self.param.props['rpn_visible'])
        self.rpn_chk.toggled.connect(self.update_param)

        gbox = QtGui.QGridLayout()
        gbox.addWidget(l_cmap, 0, 0)
        gbox.addWidget(self.combo, 0, 1)

        gbox.addWidget(l_nbr_steps, 1, 0)
        gbox.addWidget(self.nbr_steps, 1, 1)

        gbox.addWidget(l_rpn_chk, 2, 0)
        gbox.addWidget(self.rpn_chk, 2, 1)

        vbox = QtGui.QVBoxLayout()

        vbox.addLayout(gbox)
        vbox.addStretch(1.0)

        self.setLayout(vbox)

    def update_param(self, option):
        self.signal_objet_changed.emit()

class MainWindow(QtGui.QMainWindow):

    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.resize(1200, 800)
        self.setWindowTitle('Cubic Net visualization')
        icon_path = osp.abspath(osp.join(osp.dirname(__file__), 'icon.ico'))
        self.setWindowIcon(QtGui.QIcon(icon_path))

        splitter = QtGui.QSplitter(QtCore.Qt.Horizontal)
        self.canvas = Canvas()
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        self.props = ObjectWidget()
        splitter.addWidget(self.props)
        splitter.addWidget(self.canvas.native)

        self.setCentralWidget(splitter)
        self.props.signal_objet_changed.connect(self.update_view)
        self.update_view()

    def update_view(self):
        self.canvas.reset_para(self.props.nbr_steps.value(),
                               self.props.combo.currentText(),
                               self.props.rpn_chk.checkState())

class Canvas(scene.SceneCanvas):

    def __init__(self):
        scene.SceneCanvas.__init__(self, keys=None)
        self.size = 800, 600
        self.unfreeze()

        self.view = self.central_widget.add_view()
        self.radius = 2.0
        self.view.camera = 'turntable'

        self.view.camera.elevation = 19.0
        self.view.camera.center = (3.9, 3.0, 7.1)
        self.view.camera.azimuth = -90.0
        self.view.camera.scale_factor = 48
        fName = '/home/hexindong/DATASET/kittidataset/KITTI/object/training/velodyne/000169.bin'
        scans = np.fromfile(fName,dtype=np.float32).reshape(-1,4)[:,0:3]
        scatter = visuals.Markers()
        scatter.set_gl_state('translucent', depth_test=False)
        scatter.set_data(scans, edge_width=0, face_color=(1, 1, 1, 1), size=0.01, scaling=True)
        self.view.add(scatter)
        self.freeze()

    def reset_para(self, n_levels, cmap,visible):
        self.view.add(self.line_box([1,1,1,1,1,1,1,1,0]))
        a =[]
        pass
        # self.iso.set_color(cmap)
        # cl = np.linspace(-self.radius, self.radius, n_levels + 2)[1:-1]
        # self.iso.levels = cl

    def line_box(self,box,color=(0, 1, 0, 0.1)):
        p0 = np.array([box[1] - float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])
        p1 = np.array([box[1] - float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])
        p2 = np.array([box[1] + float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])
        p3 = np.array([box[1] + float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] - float(box[6]) / 2.0, ])

        p4 = np.array([box[1] - float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])
        p5 = np.array([box[1] - float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])
        p6 = np.array([box[1] + float(box[4]) / 2.0, box[2] + float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])
        p7 = np.array([box[1] + float(box[4]) / 2.0, box[2] - float(box[5]) / 2.0, box[3] + float(box[6]) / 2.0, ])

        pos = np.vstack((p0,p1,p2,p3,p0,p4,p5,p6,p7,p4,p5,p1,p2,p6,p7,p3))
        lines = visuals.Line(pos=pos, connect='strip', width=1, color=color, method='gl')

        return lines

def client_run():
    appQt = QtGui.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    appQt.exec_()
# -----------------------------------------------------------------------------


if __name__ == '__main__':
    client_run()
