__author__ = 'Sergey Matyunin'

# -*- coding: utf-8 -*-

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

from pyqtgraph.dockarea import *

list_keys_accepted = [
    QtCore.Qt.Key_1,
    QtCore.Qt.Key_2,
    QtCore.Qt.Key_3,
    QtCore.Qt.Key_4,
    QtCore.Qt.Key_5,
    QtCore.Qt.Key_6,
    QtCore.Qt.Key_7,
    QtCore.Qt.Key_8,
    QtCore.Qt.Key_9,
    QtCore.Qt.Key_0,
]

set_keys_accepted = set(list_keys_accepted)


class MyDock(Dock):
    def __init__(self, dict_key_mapping, set_keys_accepted, *args, **kwargs):
        self.dict_key_mapping_ = dict_key_mapping
        self.key_ = None

        self.set_keys_accepted = set_keys_accepted

        self.keyword_modifiers = (
            QtCore.Qt.Key_Control, QtCore.Qt.Key_Meta, QtCore.Qt.Key_Shift,
            QtCore.Qt.Key_Alt, QtCore.Qt.Key_Menu)

        super(MyDock, self).__init__(*args, **kwargs)

    def keyPressEvent(self, e):
        # print e.modifiers(), QtCore.Qt.ControlModifier,
        # print e.key()

        if e.modifiers() & QtCore.Qt.ControlModifier:
            if e.key() in self.set_keys_accepted:
                # print "Set key {} to {}".format(e.key(), self)
                self.set_key(e.key())
                return  # accept

        e.ignore()

    def set_key(self, key):
        self.key_ = key
        self.dict_key_mapping_[self.key_] = self.name()


class MyDockArea(DockArea):
    def __init__(self, list_tpl_name_image, set_keys_accepted, *args, **kwargs):
        super(MyDockArea, self).__init__(*args, **kwargs)
        self.set_keys_accepted = set_keys_accepted
        self.list_tpl_name_image = list_tpl_name_image

        self.dict_key_mapping = dict()

        first_widget = None
        self.list_dock = []
        for idx, (name, image) in enumerate(list_tpl_name_image):
            d = MyDock(name=name, size=(500, 200), dict_key_mapping=self.dict_key_mapping,
                       set_keys_accepted=self.set_keys_accepted)

            self.addDock(d, 'above')

            w = pg.PlotWidget(title="ololo" + name)
            img_item = pg.ImageItem()
            img_item.setImage(image)
            w.addItem(img_item)
            d.addWidget(w)
            if first_widget is None:
                first_widget = w
            else:
                w.plotItem.setXLink(first_widget)
                w.plotItem.setYLink(first_widget)

            try:
                d.set_key(list_keys_accepted[idx])
            except IndexError, e:
                print e

    def keyPressEvent(self, e):

        key = e.key()
        print key,
        name_dock = self.dict_key_mapping.get(key, None)
        print name_dock
        if name_dock is not None:
            t = self.docks[name_dock]
            # print t, t.container()
            try:
                t.raiseDock()
            except AttributeError, e:
                print e.message


class MyWindow2(QtGui.QMainWindow):
    def make_d1(self):
        d1 = Dock("Dock1", size=(100, 10))  ## give this dock the minimum possible size
        area = self.area
        area.addDock(d1, 'left')  ## place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)


        ## first dock gets save/restore buttons
        w1 = pg.LayoutWidget()
        label = QtGui.QLabel("""Save and load dock state""")
        saveBtn = QtGui.QPushButton('Save dock state')
        restoreBtn = QtGui.QPushButton('Restore dock state')
        restoreBtn.setEnabled(False)
        w1.addWidget(label, row=0, col=0)
        w1.addWidget(saveBtn, row=1, col=0)
        w1.addWidget(restoreBtn, row=2, col=0)
        d1.addWidget(w1)
        state = None

        def save():
            global state
            state = area.saveState()
            restoreBtn.setEnabled(True)

        def load():
            global state
            area.restoreState(state)

        saveBtn.clicked.connect(save)
        restoreBtn.clicked.connect(load)

    def __init__(self):
        super(MyWindow2, self).__init__()
        im = np.random.normal(size=(200, 100))
        im[30:50, 60:80] = 20
        im2 = np.random.normal(size=(200, 100))
        im2[60:100, 20:90] = 10
        list_tpl_name_image = [
            ("Image1", im),
            ("Image2", im2),
            ("3", im2 + im),
            ("4", im2 - im),
        ]
        area = MyDockArea(list_tpl_name_image=list_tpl_name_image, set_keys_accepted=set_keys_accepted)
        self.area = area
        self.setCentralWidget(area)
        self.resize(1000, 500)
        self.setWindowTitle('Details Viewer')

        self.make_d1()


## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        app = QtGui.QApplication([])
        win = MyWindow2()
        win.show()
        QtGui.QApplication.instance().exec_()