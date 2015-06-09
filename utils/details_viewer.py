__author__ = 'Sergey Matyunin'

# -*- coding: utf-8 -*-

import copy
import cPickle as Pickle
import os
import sys
import argparse
import logging

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import scipy.misc
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


class DockWithKeyMapping(Dock):
    def __init__(self, dict_key_mapping, set_keys_accepted, *args, **kwargs):
        super(DockWithKeyMapping, self).__init__(*args, **kwargs)

        self.key_ = None

        self.set_keys_accepted = set_keys_accepted

        self.keyword_modifiers = (
            QtCore.Qt.Key_Control, QtCore.Qt.Key_Meta, QtCore.Qt.Key_Shift,
            QtCore.Qt.Key_Alt, QtCore.Qt.Key_Menu)

        self.set_key_mapping(dict_key_mapping)

    def set_key_mapping(self, dict_key_mapping):
        self.dict_key_mapping_ = dict_key_mapping

    def keyPressEvent(self, e):
        if e.modifiers() & QtCore.Qt.ControlModifier:
            if e.key() in self.set_keys_accepted:
                self.set_key(e.key())
                return  # accept

        e.ignore()

    def set_key(self, key):
        self.key_ = key
        self.dict_key_mapping_[self.key_] = self.name()


class DockAreaWithKeyMapping(DockArea):
    _name_dict_shortcuts = 'dict_shortcuts'

    def __init__(self, dict_name_image, set_keys_accepted, *args, **kwargs):
        logger = logging.getLogger(__name__)
        super(DockAreaWithKeyMapping, self).__init__(*args, **kwargs)
        self.set_keys_accepted = set_keys_accepted
        self.list_tpl_name_image = dict_name_image

        self.dict_key_mapping = dict()

        first_widget = None
        self.list_dock = []
        for idx, (name, image) in enumerate(dict_name_image.items()):
            d = DockWithKeyMapping(name=name, size=(500, 200), dict_key_mapping=self.dict_key_mapping,
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
                logger.debug(str(e))

    def keyPressEvent(self, e):
        logger = logging.getLogger(__name__)
        if e.modifiers() & QtCore.Qt.ControlModifier:
            e.ignore()
            return

        key = e.key()

        name_dock = self.dict_key_mapping.get(key, None)

        if name_dock is not None:
            t = self.docks[name_dock]

            try:
                t.raiseDock()
            except AttributeError, e:
                logger.debug(str(e))

    def saveState(self):
        state = super(DockAreaWithKeyMapping, self).saveState()
        state[DockAreaWithKeyMapping._name_dict_shortcuts] = copy.deepcopy(self.dict_key_mapping)
        return state

    def restoreState(self, state):
        logger = logging.getLogger(__name__)
        super(DockAreaWithKeyMapping, self).restoreState(state)
        self.dict_key_mapping.clear()
        self.dict_key_mapping.update(state[DockAreaWithKeyMapping._name_dict_shortcuts])
        for k, v in self.docks.items():
            try:
                v.set_key_mapping(self.dict_key_mapping)
            except AttributeError, e:
                logger.debug(str(e))


class DViewerWindow(QtGui.QMainWindow):
    def make_d1(self):
        d1 = Dock("Dock1", size=(100, 10))  # give this dock the minimum possible size
        area = self.area
        area.addDock(d1,
                     'left')  # place d1 at left edge of dock area (it will fill the whole space since there are no other docks yet)

        w1 = pg.LayoutWidget()
        label = QtGui.QLabel("""Save and load dock state""")
        btn_save = QtGui.QPushButton('Save dock state')
        btn_restore = QtGui.QPushButton('Restore dock state')
        btn_select_file = QtGui.QPushButton('Select file')

        # btn_restore.setEnabled(False)
        w1.addWidget(label, row=0, col=0)
        w1.addWidget(btn_save, row=1, col=0)
        w1.addWidget(btn_restore, row=2, col=0)
        w1.addWidget(self.labelPathFile, row=3, col=0)
        w1.addWidget(btn_select_file, row=3, col=1)
        d1.addWidget(w1)


        def save():
            if not self.path_layout_file:
                select_file()
            if not self.path_layout_file:
                return

            state = area.saveState()
            try:
                with open(self.path_layout_file, "w") as f:
                    Pickle.dump(state, f)
            except IOError:
                msg_box = QtGui.QMessageBox()
                msg_box.setText("Failed to save file '{}'".format(self.path_layout_file))
                msg_box.exec_()
                # btn_restore.setEnabled(True)

        def load():
            if not self.path_layout_file:
                select_file()
            if not self.path_layout_file:
                return

            try:
                with open(self.path_layout_file, "r") as f:
                    state = Pickle.load(f)

                area.restoreState(state)
            except IOError:
                msg_box = QtGui.QMessageBox()
                msg_box.setText("Failed to open file '{}'".format(self.path_layout_file))
                msg_box.exec_()

        def select_file():
            self.path_layout_file = QtGui.QFileDialog.getOpenFileName(caption=QtCore.QString("Select layout file"),
                                                                      directory=os.path.abspath("."))
            self.labelPathFile.setText(self.path_layout_file)

        btn_save.clicked.connect(save)
        btn_restore.clicked.connect(load)
        btn_select_file.clicked.connect(select_file)

    def __init__(self, dict_name_image, default_path_layout_file=None):
        super(DViewerWindow, self).__init__()

        self.path_layout_file = default_path_layout_file
        area = DockAreaWithKeyMapping(dict_name_image=list_tpl_name_image, set_keys_accepted=set_keys_accepted)
        self.area = area
        self.setCentralWidget(area)
        self.resize(1000, 500)
        self.setWindowTitle('Details Viewer')

        self.labelPathFile = QtGui.QLabel(self.path_layout_file)
        self.make_d1()


# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):

        parser = argparse.ArgumentParser()
        parser.add_argument(
            'files', metavar='path', type=str,
            nargs='+', help='Path to image')
        parser.add_argument(
            '--layout', metavar='layout', type=str,
            nargs=1, help='Path to layout file')
        parser.add_argument(
            '--use_short_path', type=int, default=1, choices=(0, 1),
            help='Use full path as a key'
        )
        parameters = parser.parse_args(sys.argv[1:])

        list_tpl_name_image = {(path if parameters.use_short_path else os.path.abspath(path)): scipy.misc.imread(path)
                               for path in parameters.files}

        app = QtGui.QApplication.instance()
        if not app:
            app = QtGui.QApplication([])
        win = DViewerWindow(dict_name_image=list_tpl_name_image, default_path_layout_file="./layout.dat")
        win.show()
        QtGui.QApplication.instance().exec_()